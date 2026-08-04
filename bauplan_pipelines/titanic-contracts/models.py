import pyarrow

from typing import Annotated
from bpln_sdk import (
    TableSchema,
    TableField,
    Int64,
    Float64,
    String,
    Model,
    model,
    python,
)


class WomenByClass(TableSchema):
    """Schema for women passengers with 1st or 2nd class tickets."""

    pid:    Annotated[
        Int64,
        TableField(lineage='titanic["PassengerId"]', doc='Unique ID for a passenger'),
    ]
    tclass: Annotated[
        Int64,
        TableField(
            lineage='titanic["Pclass"]',
            doc='Ticket category as an integer, filtered to 1st or 2nd class only.'
        ),
    ]


class CabinCostSource(TableSchema):
    """Schema for cabin costs used for analysis."""

    PassengerId: Annotated[Int64, TableField(doc='Unique ID for a passenger')]
    Pclass:      Annotated[
        Int64,
        TableField(doc='Ticket category as an integer (1st, 2nd, or 3rd class).'),
    ]

    Fare:  Annotated[Float64, TableField(doc='Ticket cost (british pounds)')]
    Cabin: Annotated[String,  TableField(doc='Cabin number (C28 is on deck C)')]


class CabinCostAnalysis(TableSchema):
    """Schema for cabin costs and their average cost (by deck)."""

    pid:       Annotated[Int64,   TableField(lineage=CabinCostSource['PassengerId'])]
    tclass:    Annotated[Int64,   TableField(lineage=CabinCostSource['Pclass'])]
    Cabin:     Annotated[String,  TableField(lineage=CabinCostSource['Cabin'])]
    deck:      Annotated[String,  TableField(doc='Which deck a cabin is located on.')]
    Fare:      Annotated[Float64, TableField(lineage=CabinCostSource['Fare'])]
    mean_fare: Annotated[Float64, TableField(doc='Average cabin price by deck.')]


class AverageCabinCost(TableSchema):
    """Schema for cabin costs and their average cost (by deck)."""

    pid:       Annotated[Int64,   TableField(lineage=CabinCostAnalysis['pid'])]
    tclass:    Annotated[Int64,   TableField(lineage=CabinCostAnalysis['tclass'])]
    Fare:      Annotated[Float64, TableField(lineage=CabinCostAnalysis['Fare'])]
    Cabin:     Annotated[String,  TableField(lineage=CabinCostAnalysis['Cabin'])]
    mean_fare: Annotated[Float64, TableField(lineage=CabinCostAnalysis['mean_fare'])]
    #cabin_level: Annotated[String, TableField(lineage=CabinCostAnalysis['cabin_level'])]


class ExpensiveWomenCabins(TableSchema):
    """Schema for expensive cabins reserved by working-age women on the Titanic."""

    tclass: Annotated[Int64,  TableField(lineage=WomenByClass['tclass'])]
    Cabin:  Annotated[String, TableField(lineage=CabinCostAnalysis['Cabin'])]
    Fare:   Annotated[
        Float64,
        TableField(
            lineage=CabinCostAnalysis['Fare'],
            doc='Ticket cost filtered to tickets that cost more than average (by deck).'
        ),
    ]


@python('3.13', pip={'polars': '1.37'})
@model(materialization_strategy='REPLACE')
def g(
    adult_lodgings: Annotated[
        pyarrow.Table,
        Model(
            'titanic',
            projection_schema=CabinCostSource,
            filter='Age >= 18 AND Age <= 65',
        ),
    ],
) -> Annotated[pyarrow.Table, CabinCostAnalysis]:
    """Analyze the cost by deck for working-age adults."""

    import polars as pl
    result_df = (
        pl.DataFrame(adult_lodgings)
           # Get the cabin level ("C" in "C28")
          .with_columns(deck=(pl.col('Cabin').str.slice(0, 1)))
           # Calculate average Fare by deck using a window function
          .with_columns(mean_fare=(pl.col('Fare').mean().over('deck')))
          .select(
             pl.col('PassengerId').alias('pid'),
             pl.col('Pclass').alias('tclass'),
             'Cabin', 'deck', 'Fare', 'mean_fare',
           )
    )

    print(result_df)
    return result_df.to_arrow()


@python('3.13', pip={'polars': '1.37', 'bpln-pysdk-internal': '0.0.2'})
@model(materialization_strategy='REPLACE')
def h(
    f_data: Annotated[
        pyarrow.Table,
        Model('f', projection_schema=WomenByClass)
    ],
    g_data: Annotated[
        pyarrow.Table,
        Model('g', projection_schema=AverageCabinCost)
    ],
) -> Annotated[pyarrow.Table, ExpensiveWomenCabins]:
    """Determine expensive cabins reserved by working-age women."""

    import polars as pl
    result_df = (
        pl.DataFrame(f_data)
          .join(pl.DataFrame(g_data), on=('pid', 'tclass'))
          .filter((pl.col('Fare') > pl.col('mean_fare')))
          .select('tclass', 'Cabin', 'Fare')
    )

    print(result_df)
    return result_df.to_arrow()
