from feast import Entity, Feature, FeatureView, ValueType, Field, FileSource
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
from feast.types import Float32, Int64
from datetime import timedelta


# Define an entity (e.g., "house_id" for each house)
house = Entity(name="house_id", join_keys=["house_id"], value_type=ValueType.INT64, description="Unique ID for each house")

housing_source = FileSource(
    path="..\\data\\house_features.parquet",
    event_timestamp_column="event_timestamp",  # Optional: add timestamp if your data includes it
)

pg_source = PostgreSQLSource(
    name="house_features_sql",
    query="SELECT * FROM house_features_sql",
    timestamp_field="event_timestamp",
    #created_timestamp_column="event_time"
)

# Define a Feature View
house_features = FeatureView(
    name="house_features",
    entities=[house],
    ttl=timedelta(seconds=86400 * 10),  # Time-to-live for the features
    schema=[
        Field(name="area", dtype=Float32),
        Field(name="bedrooms", dtype=Float32),
        Field(name="bathrooms", dtype=Float32),
        Field(name="stories", dtype=Float32),
        Field(name="mainroad", dtype=Int64),
        Field(name="guestroom", dtype=Int64),
        Field(name="basement", dtype=Int64),
        Field(name="hotwaterheating", dtype=Int64),
        Field(name="airconditioning", dtype=Int64),
        Field(name="parking", dtype=Float32),
        Field(name="prefarea", dtype=Int64),
        Field(name="furnishingstatus", dtype=Int64),
    ],
    online=True,  # Indicates that the feature view is accessible in the online store
    source= pg_source,  # We'll load data programmatically
)