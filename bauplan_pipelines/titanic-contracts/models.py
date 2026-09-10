from typing import Annotated

import pyarrow
import bauplan

from bauplan import TableField, Int64, Float64, String


class WomenByClass(bauplan.TableSchema):
    """Schema for women passengers with 1st or 2nd class tickets."""

    pid: Annotated[Int64 | None, TableField(doc="Unique ID for a passenger")]
    tclass: Annotated[
        Int64 | None,
        TableField(
            doc="Ticket category as an integer, filtered to 1st or 2nd class only."
        ),
    ]


class CabinCostSource(bauplan.TableSchema):
    """Schema for cabin costs used for analysis."""

    PassengerId: Annotated[Int64 | None, TableField(doc="Unique ID for a passenger")]
    Pclass: Annotated[
        Int64 | None,
        TableField(doc="Ticket category as an integer (1st, 2nd, or 3rd class)."),
    ]

    Fare: Annotated[Float64 | None, TableField(doc="Ticket cost (british pounds)")]
    Cabin: Annotated[String | None, TableField(doc="Cabin number (C28 is on deck C)")]


class CabinCostAnalysis(bauplan.TableSchema):
    """Schema for cabin costs and their average cost (by deck)."""

    pid: Annotated[Int64 | None, TableField(lineage=CabinCostSource["PassengerId"])]
    tclass: Annotated[Int64 | None, TableField(lineage=CabinCostSource["Pclass"])]
    Cabin: Annotated[String | None, TableField(lineage=CabinCostSource["Cabin"])]
    deck: Annotated[String | None, TableField(doc="Which deck a cabin is located on.")]
    Fare: Annotated[Float64 | None, TableField(lineage=CabinCostSource["Fare"])]
    mean_fare: Annotated[Float64 | None, TableField(doc="Average cabin price by deck.")]


class AverageCabinCost(bauplan.TableSchema):
    """Schema for cabin costs and their average cost (by deck)."""

    pid: Annotated[Int64 | None, TableField(lineage=CabinCostAnalysis["pid"])]
    tclass: Annotated[Int64 | None, TableField(lineage=CabinCostAnalysis["tclass"])]
    Fare: Annotated[Float64 | None, TableField(lineage=CabinCostAnalysis["Fare"])]
    Cabin: Annotated[String | None, TableField(lineage=CabinCostAnalysis["Cabin"])]
    mean_fare: Annotated[
        Float64 | None, TableField(lineage=CabinCostAnalysis["mean_fare"])
    ]


class ExpensiveWomenCabins(bauplan.TableSchema):
    """Schema for expensive cabins reserved by working-age women on the Titanic."""

    pid: Annotated[Int64 | None, TableField(lineage=WomenByClass["pid"])]
    tclass: Annotated[Int64 | None, TableField(lineage=WomenByClass["tclass"])]
    Cabin: Annotated[String | None, TableField(lineage=CabinCostAnalysis["Cabin"])]
    Fare: Annotated[
        Float64 | None,
        TableField(
            lineage=CabinCostAnalysis["Fare"],
            doc="Ticket cost filtered to tickets that cost more than average (by deck).",
        ),
    ]


@bauplan.python("3.13", pip={"polars": "1.37"})
@bauplan.model(materialization_strategy="REPLACE")
def g(
    adult_lodgings: Annotated[
        pyarrow.Table,
        bauplan.Model(
            "titanic",
            projection_schema=CabinCostSource,
            filter="Age >= 18 AND Age <= 65",
        ),
    ],
) -> Annotated[pyarrow.Table, CabinCostAnalysis]:
    """Analyze the cost by deck for working-age adults."""

    import polars as pl  # ty: ignore[unresolved-import]

    result_table = (
        pl.DataFrame(adult_lodgings)
        # Get the cabin level ("C" in "C28")
        .with_columns(deck=(pl.col("Cabin").str.slice(0, 1)))
        # Calculate average Fare by deck using a window function
        .with_columns(mean_fare=(pl.col("Fare").mean().over("deck")))
        .select(
            pl.col("PassengerId").alias("pid"),
            pl.col("Pclass").alias("tclass"),
            "Cabin",
            "deck",
            "Fare",
            "mean_fare",
        )
        .to_arrow()
    )

    print(result_table)
    return result_table


@bauplan.python("3.13", pip={"polars": "1.37"})
@bauplan.model(materialization_strategy="REPLACE")
def h(
    f_data: Annotated[
        pyarrow.Table, bauplan.Model("f", projection_schema=WomenByClass)
    ],
    g_data: Annotated[
        pyarrow.Table, bauplan.Model("g", projection_schema=AverageCabinCost)
    ],
) -> Annotated[pyarrow.Table, ExpensiveWomenCabins]:
    import polars as pl  # ty: ignore[unresolved-import]

    result_table = (
        pl.DataFrame(f_data)
        .join(pl.DataFrame(g_data), on=("pid", "tclass"))
        .filter((pl.col("Fare") > pl.col("mean_fare")))
        .select("pid", "tclass", "Cabin", "Fare")
        .to_arrow()
    )

    print(result_table)
    return result_table
