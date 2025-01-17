import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sqlalchemy import create_engine
import logging

from House_price_prediction import *
from monitoring.evidently_monitoring import *
from model.house_model import HouseModel

from evidently.collector.config import ReportConfig
from evidently.collector.client import CollectorClient
from evidently.collector.config import CollectorConfig
from evidently.collector.config import IntervalTrigger
from evidently.collector.config import ReportConfig
from evidently.collector.config import RowsCountTrigger

logging.basicConfig(   
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    force=True
)

WORKSPACE = "monitoring workspace"
PROJECT = "live dashboard"
COLLECTOR_ID = "house_ev"
COLLECTOR_QUAL_ID = "house_ev_qual"
COLLECTOR_TGT_ID = "house_ev_tgt"
COLLECTOR_REG_ID = "house_ev_reg"
COLLECTOR_TEST_ID = "house_ev_test"


class Dashboard():
    def __init__(self):
        self.monitoring = Monitoring(DataDriftReport())
        self.house = HousePricePrediction()     
        self.house_model = HouseModel()
        self.client = CollectorClient("http://localhost:8001")
        self.ws = None
        self.reference = None
        self.column_mapping = ColumnMapping()
        self.start_date = datetime.now() - timedelta(days=20)
        
    def get_db_connection(self):
        connstr = 'postgresql+psycopg2://postgres:root@localhost:5432/feast_offline'
        engine = create_engine(connstr)
        return engine
    
    def get_reference_and_current_data(self):
        feature_list=[
            "house_features:area",
            "house_features:bedrooms",
            "house_features:mainroad"
        ]
        logging.info("GEt data enter")

        store = self.house.get_feature_store()
        features = self.house.get_historical_features()
        entity_df_ref = pd.DataFrame(features["house_id"])
        reference = self.house.get_online_features(store, entity_df_ref)
        engine = self.get_db_connection()
        entity_df_cur = pd.read_sql(str.format("select house_id, price from public.house_target_sql where event_timestamp >= '{0}'", self.start_date.strftime(r'%Y-%m-%d %H:%M:%S')), con=engine)
        current = self.house.get_online_features(store, pd.DataFrame(entity_df_cur["house_id"]))
        lr_model = self.house_model.load_model()        
        
        #Adding target and prediction to data
        reference = reference.drop("house_id", axis=1)
        reference = reference[lr_model.feature_names_in_]
        reference["prediction"] = self.house_model.predict(reference)
        reference["price"] = features["price"]
        logging.info(reference)
        
        current = current.drop("house_id", axis=1)
        current = current[lr_model.feature_names_in_]
        current["prediction"] = self.house_model.predict(current)
        current["price"] = entity_df_cur["price"]
        logging.info(current)

        return reference, current
     
    def get_project(self):
        self.ws = self.monitoring.create_workspace(WORKSPACE)
        project = self.monitoring.search_or_create_project(PROJECT)
        return project

    def create_reports(self):
        self.reference, current = self.get_reference_and_current_data()
        self.column_mapping = ColumnMapping()
        self.column_mapping.target = "price"
        self.column_mapping.prediction = "prediction"
        #Data drift report
        print(self.monitoring.current_strategy)
        drift_report = self.monitoring.execute_strategy(self.reference, current, self.ws)
        rep_config = ReportConfig.from_report(drift_report)

        #Data quality report
        self.monitoring.set_strategy = DataQualityReport()
        qual_report = self.monitoring.execute_strategy(self.reference, current, self.ws)
        qual_rep_config = ReportConfig.from_report(qual_report)
        
        #Target drift report
        self.monitoring.set_strategy = TargetDriftReport()
        target_report = self.monitoring.execute_strategy(self.reference, current, self.ws)
        target_rep_config = ReportConfig.from_report(target_report)

        self.monitoring.set_strategy = RegressionReport()
        reg_report = self.monitoring.execute_strategy(self.reference, current, self.ws, self.column_mapping)    
        reg_rep_config = ReportConfig.from_report(reg_report)


        #Data drfit test report
        self.monitoring.set_strategy = DataDriftTestReport()
        print(self.monitoring.current_strategy)
        test_report = self.monitoring.execute_strategy(self.reference, current, self.ws)
        test_rep_config = ReportConfig.from_test_suite(test_report)
        
        logging.info("All reports are created")
        return rep_config, qual_rep_config, target_rep_config, reg_rep_config, test_rep_config

    def create_live_dashboard(self, project: evidently.ui.base.Project):
         #Create dashboard panels
        self.monitoring.add_dashboard_panel(
            project, panel_type="Counter", 
            title = "House price Monitoring dashboard",
            tags = [],  
            metric_id = None,
            field_path = "",
            legend = "",
            text = "",
            agg = CounterAgg.NONE,
            size = WidgetSize.FULL
        )

        self.monitoring.add_dashboard_panel(
            project, panel_type="Counter", 
            title = "Number of columns",
            tags = [],  
            metric_id = "DatasetDriftMetric",
            field_path = "number_of_columns",
            legend = "",
            text = "",
            agg = CounterAgg.LAST,
            size = WidgetSize.HALF
        )

        self.monitoring.add_dashboard_panel(
            project, panel_type="Counter", 
            title = "Number of drifted columns",
            tags = [],  
            metric_id = "DatasetDriftMetric",
            field_path = "number_of_drifted_columns",
            legend = "",
            text = "",
            agg = CounterAgg.LAST,
            size = WidgetSize.HALF
        )
        
        self.monitoring.add_dashboard_panel(
            project, panel_type="Counter", 
            title = "Target column drift score",
            tags = [],  
            metric_id = "ColumnDriftMetric",
            field_path = "drift_score",
            legend = "",
            text = "",
            agg = CounterAgg.LAST,
            size = WidgetSize.HALF
        )
        
        self.monitoring.add_dashboard_panel(
            project, panel_type="Counter", 
            title = "Number of missing values - Current",
            tags = [],  
            metric_id = "DatasetMissingValuesMetric",
            field_path = "current.number_of_missing_values",
            metric_args = {},
            legend = "Current - missing values",
            size = WidgetSize.HALF,
            agg = CounterAgg.LAST,
            text = ""
        )

        self.monitoring.add_dashboard_panel(
            project, panel_type="Plot", 
            title = "Share of drifted columns",
            tags = [],  
            metric_id = "DatasetDriftMetric",
            field_path = "share_of_drifted_columns",
            metric_args = {},
            legend = "share",
            plot_type = PlotType.LINE,
            size = WidgetSize.HALF,
            agg = CounterAgg.SUM
        )

        self.monitoring.add_dashboard_panel(
            project, panel_type="MultiPlot", 
            title = "R2 score - Current vs Reference",
            tags = [],  
            metric_id = "RegressionQualityMetric",
            field_path = "current.r2_score",
            metric_args = {},
            legend = "R2 Current",
            metric_id_2 = "RegressionQualityMetric",
            field_path_2 = "reference.r2_score",
            metric_args_2 = {},
            legend_2 = "Reference R2",
            plot_type = PlotType.LINE,
            size = WidgetSize.HALF,
            agg = CounterAgg.SUM
        )
        
        self.monitoring.add_dashboard_panel(
            project, panel_type="MultiPlot", 
            title = "MAE score - Current vs Reference",
            tags = [],  
            metric_id = "RegressionQualityMetric",
            field_path = "current.mean_abs_error",
            metric_args = {},
            legend = "MAE",
            metric_id_2 = "RegressionQualityMetric",
            field_path_2 = "reference.mean_abs_error",
            metric_args_2 = {},
            legend_2 = "Reference MAE",
            plot_type = PlotType.LINE,
            size = WidgetSize.HALF,
            agg = CounterAgg.SUM
        )
        
    
    def configure_collector(self):
        project = self.get_project()
        rep_config, qual_rep_config, target_rep_config, reg_rep_config, test_rep_config = self.create_reports()
        self.create_live_dashboard(project)

        conf = CollectorConfig(
            trigger = IntervalTrigger(interval=30),
            report_config = rep_config,
            project_id = str(project.id)
        )
        self.client.create_collector(id=COLLECTOR_ID, collector=conf)

        conf_qual = CollectorConfig(
            trigger = IntervalTrigger(interval=30),
            report_config = qual_rep_config,
            project_id = str(project.id)
        )
        self.client.create_collector(id=COLLECTOR_QUAL_ID, collector=conf_qual)

        conf_target = CollectorConfig(
            trigger = IntervalTrigger(interval=30),
            report_config = target_rep_config,
            project_id = str(project.id)
        )
        self.client.create_collector(id=COLLECTOR_TGT_ID, collector=conf_target)

        conf_reg = CollectorConfig(
            trigger = IntervalTrigger(interval=30),
            report_config = reg_rep_config,
            project_id = str(project.id)
        )
        self.client.create_collector(id=COLLECTOR_REG_ID, collector=conf_reg)

        test_conf = CollectorConfig(
            trigger=RowsCountTrigger(interval=30), 
            report_config=test_rep_config,
            project_id=str(project.id)
        )
        self.client.create_collector(id=COLLECTOR_TEST_ID, collector=test_conf)
        
        self.client.set_reference(id=COLLECTOR_ID, reference=self.reference)
        self.client.set_reference(id=COLLECTOR_TGT_ID, reference=self.reference)
        self.client.set_reference(id=COLLECTOR_QUAL_ID, reference=self.reference)
        self.client.set_reference(id=COLLECTOR_REG_ID, reference=self.reference)
        self.client.set_reference(id=COLLECTOR_TEST_ID, reference=self.reference)

    def send_data_to_collector(self):
        self.reference , current = self.get_reference_and_current_data()
        self.client.send_data(COLLECTOR_ID, current)
        self.client.send_data(COLLECTOR_TGT_ID, current)
        self.client.send_data(COLLECTOR_REG_ID, current)
        self.client.send_data(COLLECTOR_QUAL_ID, current)
        self.client.send_data(COLLECTOR_TEST_ID, current)

if __name__ == "__main__":
    os.chdir("/home/edwin/git/mlops-open-source-tools/")
    dashboard = Dashboard()
    if not os.path.exists(os.path.join(os.getcwd(), WORKSPACE)) or \
        len(Workspace.create(os.path.join(os.getcwd(), WORKSPACE)).search_project(PROJECT)) == 0:
        dashboard.configure_collector()
    dashboard.send_data_to_collector()
    print("Triggered monitoring")
    logging.info("Triggered monitoring")
    #dashboard.client.send_data(COLLECTOR_ID, current)
    #dashboard.client.send_data(COLLECTOR_TEST_ID, current)


