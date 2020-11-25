"""Contents of hello_kedro.py"""
# prerequisite
# kedro jupyter convert notebooks/generationCopy1.ipynb
from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import node, Pipeline
from kedro.runner import SequentialRunner
from src.kedro_bioimage.nodes.generationCopy1 import generation

# Prepare a data catalog
data_catalog = DataCatalog({"example_data": MemoryDataSet()})

# Prepare first node
def return_greeting():
    return "Hello"


return_greeting_node = node(
    return_greeting, inputs=None, outputs="my_salutation"
)

# Prepare second node
def join_statements(greeting):
    return f"{greeting} Kedro!"


join_statements_node = node(
    join_statements, inputs="my_salutation", outputs="my_message"
)

#

# adder_node = node(
#     func=add, inputs=["a", "b"], outputs="sum"
# )

generation_node = node(
    generation, inputs=["num_samples"], outputs=None
)

# Assemble nodes into a pipeline
#pipeline = Pipeline([return_greeting_node, join_statements_node])
#pipeline = Pipeline([return_greeting_node])
pipeline = Pipeline([generation_node])
# Create a runner to run the pipeline
runner = SequentialRunner()

# Run the pipeline
#print(runner.run(pipeline, data_catalog))
#print(runner.run(pipeline, DataCatalog(dict(a=2,b=3))))


#io = DataCatalog(dict(a=MemoryDataSet(),b=MemoryDataSet()))
io = DataCatalog(dict(num_samples=MemoryDataSet()))
io.save("num_samples",1)
# print(adder_node.run(dict(a=2, b=3)))
print(runner.run(pipeline, io))
