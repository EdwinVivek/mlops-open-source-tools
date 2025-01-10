import os
import sys
sys.path.append(os.getcwd())

from monitoring.evidently_monitoring import *
from airflow.monitor_drift import MonitorDrift

from evidently.collector.config import ReportConfig
from evidently.collector.client import CollectorClient
from evidently.collector.config import CollectorConfig
from evidently.collector.config import IntervalTrigger
from evidently.collector.config import ReportConfig
from evidently.collector.config import RowsCountTrigger

WORKSPACE = "monitoring workspace"
PROJECT = "live dashboard"
COLLECTOR_ID = "house_ev"
COLLECTOR_TEST_ID = "house_ev_test"


class Dashboard():
    def __init__(self):
        self.monitoring = Monitoring(DataDriftReport())
        self.drift_monitor = MonitorDrift()
        self.client = CollectorClient("http://localhost:8001")
        self.ws = None
        self.reference = None

    def get_project(self):
        self.ws = self.monitoring.create_workspace(WORKSPACE)
        project = self.monitoring.search_or_create_project(PROJECT)
        return project

    def create_reports(self):
        self.reference, current = self.drift_monitor.get_reference_and_current_data()
        
        #Data drift report
        print(self.monitoring.current_strategy)
        drift_report = self.monitoring.execute_strategy(self.reference, current, self.ws)
        rep_config = ReportConfig.from_report(drift_report)


        #Data drfit test report
        self.monitoring.set_strategy = DataDriftTestReport()
        print(self.monitoring.current_strategy)
        test_report = self.monitoring.execute_strategy(self.reference, current, self.ws)
        test_rep_config = ReportConfig.from_test_suite(test_report)
        return rep_config, test_rep_config

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
            title = "Number of drifted columns",
            tags = [],  
            metric_id = "DatasetDriftMetric",
            field_path = "Drifted Columns",
            legend = "",
            text = "",
            agg = CounterAgg.LAST,
            size = WidgetSize.HALF
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

    
    def configure_collector(self):
        project = self.get_project()
        rep_config, test_rep_config = self.create_reports()
        self.create_live_dashboard(project)

        conf = CollectorConfig(
            trigger = IntervalTrigger(interval=30),
            report_config = rep_config,
            project_id = str(project.id)
        )
        self.client.create_collector(id=COLLECTOR_ID, collector=conf)


        test_conf = CollectorConfig(
            trigger=RowsCountTrigger(interval=30), 
            report_config=test_rep_config,
            project_id=str(project.id)
        )
        self.client.create_collector(id=COLLECTOR_TEST_ID, collector=test_conf)
        
        self.client.set_reference(id=COLLECTOR_ID, reference=self.reference)
        self.client.set_reference(id=COLLECTOR_TEST_ID, reference=self.reference)

    def send_data_to_collector(self):
        _ , current = self.drift_monitor.get_reference_and_current_data()
        self.client.send_data(COLLECTOR_ID, current)
        self.client.send_data(COLLECTOR_TEST_ID, current)

if __name__ == "__main__":
    dashboard = Dashboard()
    if not os.path.exists(os.path.join(os.getcwd(), WORKSPACE)) or \
        len(Workspace.create(os.path.join(os.getcwd(), WORKSPACE)).search_project(PROJECT)) == 0:
        dashboard.configure_collector()
    dashboard.send_data_to_collector()
    print("Triggered monitoring")
    #dashboard.client.send_data(COLLECTOR_ID, current)
    #dashboard.client.send_data(COLLECTOR_TEST_ID, current)


