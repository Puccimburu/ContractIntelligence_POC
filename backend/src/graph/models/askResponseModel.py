from typing import List, Literal, Union, Optional, Annotated
from pydantic import BaseModel, Field, RootModel

#
# I. Define nested models first, which will be used in the main component models.
#

class Point(BaseModel):
    """
    A model to represent a single data point in a scatter chart,
    which has 'x' and 'y' coordinates.
    """
    x: Union[int, float]
    y: Union[int, float]

class Dataset(BaseModel):
    """
    A model for a single dataset within a chart.
    It can contain various properties to define its appearance and data.
    """
    label: str
    # The data can be a list of numbers (for most charts) or a list of Point objects (for scatter charts).
    data: List[Union[int, float, Point]]
    # Optional fields for styling the chart dataset.
    backgroundColor: Optional[Union[str, List[str]]] = None
    borderColor: Optional[str] = None
    fill: Optional[bool] = None
    hoverOffset: Optional[int] = None
    pointBackgroundColor: Optional[str] = None
    pointBorderColor: Optional[str] = None
    pointHoverBackgroundColor: Optional[str] = None
    pointHoverBorderColor: Optional[str] = None

class ChartData(BaseModel):
    """
    A model representing the complete data object for a chart,
    including labels and multiple datasets.
    """
    # Labels are optional because scatter charts do not require them.
    labels: Optional[List[str]] = None
    datasets: List[Dataset]

#
# II. Define the main component models, one for each "type".
#

class TextComponent(BaseModel):
    """A model for a simple text block."""
    type: Literal["text"]
    content: str

class TableComponent(BaseModel):
    """A model for tabular data with headers and rows."""
    type: Literal["table"]
    headers: List[str]
    rows: List[List[str]]

class ChartComponent(BaseModel):
    """A model for a chart, which can be of various types."""
    type: Literal["chart"]
    # The chartType is restricted to the specific types found in the JSON data.
    chartType: Literal["bar", "line", "pie", "doughnut", "radar", "scatter"]
    data: ChartData

class HtmlComponent(BaseModel):
    """A model for a block of raw HTML content."""
    type: Literal["html"]
    content: str

#
# III. Create a Union of all component types and a Root Model.
#

# Use Annotated and a Field discriminator to tell Pydantic how to choose
# the correct model from the Union based on the 'type' field.
AnyComponent = Annotated[
    Union[TextComponent, TableComponent, ChartComponent, HtmlComponent],
    Field(discriminator="type")
]

# The RootModel represents the top-level structure, which is a list of components.
DashboardModel = RootModel[List[AnyComponent]]
