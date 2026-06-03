# --- Azure Authentication ---

# Azure Authentication Question 1
# In a comment block, answer: when you run a Python script locally that uses DefaultAzureCredential, what does it rely on to authenticate? What command must you have run first, and how does DefaultAzureCredential know to use it?

# A: It relies on command az login. When we create a DefaultAzureCredential instance, it tries a sequence of authentication methods in order, stopping at the first one that works. The sequence includes environment variables, managed identity (for VMs and containers running in Azure), az login session, and others. So when we run az login, DefaultAzureCredential can pick up that session automatically.

# Azure Authentication Question 2
# In a comment block, answer: why can't a deployed pipeline (running on an Azure VM or container) use az login for authentication? What does it use instead, and why does the same Python code work without changes?

# A: Cloud platforms handle this through an identity-based model. This identity, which is called service principal, is used by a cloud service to communicate with other cloud services. That way when this cloud service runs the python code, it will work with successful permission because of its given service principal identity.

# Azure Authentication Question 3
# You run a script that creates a DefaultAzureCredential and immediately gets an AuthenticationError. In a comment block, describe the two most likely causes and how you would diagnose each.

# A: Two common cases are 1) environment credentials are incorrectly set. To diagonise check env credentials for correctness. 2) az login is not setup properly, no managed identity found through CLI. To diagonise run command az account show.

# Blob Storage

# Blob Storage Question 1
# In a comment block, describe the three-level hierarchy of Azure Blob Storage in your own words. Give a concrete analogy that maps each level to something familiar (a filesystem, a filing cabinet, etc.).

# A: So the hierarchy looks like this: Storage Account → Container → Blob (Building → Filing Cabinet → File), the concreate analogy wouuld ne russia dolls stacked in each other. Storage Account is a top-level namespace, outer layer. In order to access inside, you need account name, credentials. Container is a logical grouping of blobs within the account. It keeps blobs inside isolated. Blob is the individual piece of data (object) you store—could be a text file, image, video, etc. Blobs live inside containers.

# Blob Storage Question 2
# For each scenario below, write one sentence in a comment block saying whether you would use Blob Storage or a relational database (like Azure SQL), and why.

# A REST API returns a JSON payload each hour. You need to store the raw responses for reprocessing later. -- A: Use Blob Storage – it’s ideal for storing raw, unstructured files (like JSON) that you’ll read back later.

# Your pipeline produces a table of 50 million customer transactions that your analytics team queries by date range and customer ID every day. -- A: Use a relational database – SQL engines (e.g. Azure SQL) excel at indexing and querying large tabular datasets by key or date.

# A computer vision model produces image embeddings as NumPy arrays. You need to save them between pipeline runs. -- A: Use Blob Storage – you can serialize each array (e.g. .npy or .npz) as an unstructured object and load it back into your pipeline.

# Blob Storage Question 3
# Write a function list_container(container_client) that prints the name and size (in bytes) of every blob in the container, one per line. The function should take a ContainerClient object as its only argument and return nothing.


from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
container = ContainerClient(
    account_url="https://merimctd2026sa.blob.core.windows.net",
    container_name="pipeline-data",
    credential=credential
)


def list_container(container_client):

    blobs = container_client.list_blobs()

    for blob in blobs:

        name = blob.name
        size = blob.size

        print(f"Name:{name} and size: {size}")


list_container(container)


# Blob Storage Question 4
# Write a function upload_text(container_client, blob_name, text) that encodes a Python string as UTF-8 and uploads it as a blob, overwriting any existing blob with the same name. The function should take a ContainerClient, a blob name string, and a text string, and return nothing.

def upload_text(container_client, blob_name, text):
    container_client.upload_blob(
        name=blob_name,
        data=text.encode("utf-8"),
        overwrite=True
    )


upload_text(container, "message.txt", "great")
