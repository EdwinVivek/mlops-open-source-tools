import evidently
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
from evidently.metrics import ColumnSummaryMetric, ColumnQuantileMetric, ColumnDriftMetric
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset, RegressionTestPreset, DataDriftTestPreset
from evidently.tests import TestNumberOfDriftedColumns
from evidently.ui.workspace import Workspace
from evidently.ui.workspace import WorkspaceBase, RemoteWorkspace
from evidently.ui.dashboards import DashboardPanelCounter, DashboardPanelPlot, ReportFilter, PanelValue, PlotType, CounterAgg, TestSuitePanelType, DashboardPanelTestSuite
from evidently.renderers.html_widgets import WidgetSize

from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime


class GenerateReport(ABC):
    @abstractmethod
    def create_report(self, workspace, project, reference, current, column_mapping):
        pass

class DataDriftReport(GenerateReport):
    def create_report(self, workspace, project, reference, current, column_mapping):
        data_drift_report = Report(metrics=[DataDriftPreset()], timestamp=datetime.now(), tags=["drift_rep"])
        data_drift_report.run(reference_data=reference, current_data=current)
        workspace.add_report(project_id=project.id, report=data_drift_report)
        return data_drift_report

class TargetDriftReport(GenerateReport):
    def create_report(self, workspace, project, reference, current, column_mapping):
        target_drift_report = Report(metrics=[TargetDriftPreset()], tags=["targetdrift_rep"])
        target_drift_report.run(reference_data=reference, current_data=current, column_mapping = column_mapping)
        workspace.add_report(project_id=project.id, report=target_drift_report)
        return target_drift_report

class DataQualityReport(GenerateReport):
    def create_report(self, workspace, project, reference, current, column_mapping):
        data_quality_report = Report(metrics=[DataQualityPreset()], tags=["quality_rep"])
        data_quality_report.run(reference_data=reference, current_data=current)
        workspace.add_report(project_id=project.id, report=data_quality_report)
        return data_quality_report

class RegressionReport(GenerateReport):
    def create_report(self, workspace, project, reference, current, column_mapping):
        regression_performance_report = Report(metrics=[RegressionPreset()], tags=["regression_rep"])
        regression_performance_report.run(reference_data=reference, current_data=current, column_mapping = column_mapping)
        workspace.add_report(project_id=project.id, report=regression_performance_report)
        return regression_performance_report

class ClassificationReport(GenerateReport):
    def create_report(self, workspace, project, reference, current, column_mapping):
        pass

class DataDriftTestReport(GenerateReport):
    def create_report(self, workspace, project, reference, current, column_mapping):
        data_drift_test_suite = TestSuite(tests=[DataDriftTestPreset(), TestNumberOfDriftedColumns()])
        data_drift_test_suite.run(reference_data=reference, current_data=current)
        workspace.add_test_suite(project_id=project.id, test_suite= data_drift_test_suite)
        return data_drift_test_suite
 


class Monitoring:
    def __init__(self, strategy = DataDriftReport()):
        self._strategy = strategy
        self._workspace = None
        self._project = None

    def create_workspace(self, name:str):
        self._workspace = Workspace.create(name)
        return self._workspace
    
    def search_or_create_project(self, project_name:str, workspace: WorkspaceBase = None):
        if(self._workspace is None):
            self._workspace = workspace
        project_list = self._workspace.search_project(project_name=project_name)
        if(len(project_list) == 0):
            self._project = self._workspace.create_project(project_name)
        else:
            self._project = project_list[0]
        return self._project

    @property
    def current_strategy(self):
        return self._strategy
        
    @current_strategy.setter
    def set_strategy(self, strategy: DataDriftReport):
        self._strategy = strategy

    def execute_strategy(self, reference: pd.DataFrame, current: pd.DataFrame, workspace: WorkspaceBase = None, column_mapping : ColumnMapping = None):
        if(self._workspace is None):
            self._workspace = workspace
        report = self._strategy.create_report(self._workspace, self._project, reference, current, column_mapping)
        print("Report Created successfully!!")
        return report

    def add_dashboard_panel(self, project: evidently.ui.base.Project, panel_type: str, **kwargs):
        match panel_type:
            case "Counter":
                project.dashboard.add_panel(
                    DashboardPanelCounter(
                        title=kwargs["title"],
                        filter=ReportFilter(metadata_values={}, tag_values=kwargs["tags"]),
                        value=PanelValue(
                            metric_id=kwargs["metric_id"],
                            field_path=kwargs["field_path"],
                            legend=kwargs["legend"],
                        ),
                        text=kwargs["text"],
                        agg=kwargs["agg"],
                        size=kwargs["size"],
                    )
                )

            case "Plot":
                project.dashboard.add_panel(
                    DashboardPanelPlot(
                        title=kwargs["title"],
                        filter=ReportFilter(metadata_values={}, tag_values=[]),
                        values=[
                            PanelValue(
                                metric_id=kwargs["metric_id"],
                                metric_args=kwargs["metric_args"],
                                field_path=kwargs["field_path"],
                                legend=kwargs["legend"]
                            ),
                        ],
                        plot_type=kwargs["plot_type"],
                        size=kwargs["size"]
                    )
                )
                
            case "MultiPlot":
                project.dashboard.add_panel(
                    DashboardPanelPlot(
                        title=kwargs["title"],
                        filter=ReportFilter(metadata_values={}, tag_values=[]),
                        values=[
                            PanelValue(
                                metric_id=kwargs["metric_id"],
                                metric_args=kwargs["metric_args"],
                                field_path=kwargs["field_path"],
                                legend=kwargs["legend"]
                            ),
                            PanelValue(
                                metric_id=kwargs["metric_id_2"],
                                metric_args=kwargs["metric_args_2"],
                                field_path=kwargs["field_path_2"],
                                legend=kwargs["legend_2"]
                            ),
                        ],
                        plot_type=kwargs["plot_type"],
                        size=kwargs["size"]
                    )
                )

            case "TestSuite":
                project.dashboard.add_panel(
                    DashboardPanelTestSuite(
                        title="All tests: detailed",
                        filter=ReportFilter(metadata_values={}, tag_values=[], include_test_suites=True),
                        size=WidgetSize.FULL,
                        panel_type=TestSuitePanelType.DETAILED,
                        time_agg="1H",
                    )
                )

            case _:
                print("Specified panel type not defined!")
                
        project.save()
        print(f"Panel {panel_type} created!!")

    def delete_dashboard(self, project: evidently.ui.base.Project):
        project.dashboard.panels = []
        project.save()
        print("Panels deleted!!")






