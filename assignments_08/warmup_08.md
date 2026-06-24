Cloud Concepts Question 1

Q: What is the core economic model of cloud computing, and how does it differ from owning your own servers?

A: Cloud providers such as AWS, GCP, and Azure rent computing system out to their clients. These cloud providers own and run huge data centers all around the world in order to provide this computing, because it is somewhere else, we call it in cloud. 

Cloud Concepts Question 2
Q: What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

A: Scaling here means those rented machines scale up and down based on the traffic in two ways: horisonal and vertical. Vertical scaling is when making one machine bigger (more RAM, CPU, GPU). Horizontal scaling is when adding more machines and splitting the work across them. 


Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch. A: -- Horizontal, meaning we need to spin up more machines to handle such traffic
A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM. A: -- Vertical, meaning more GPU and RAM to make hamdle more work

A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines. A: -- Horizontal, as it indicates spliting across machines, so we need more machines to process more files.


Cloud Concepts Question 3

Before writing your definitions, classify each item in the list below as IaaS, PaaS, or SaaS. One sentence of reasoning is enough for each.

Gmail -- SaaS
Azure Virtual Machines -- IaaS
Azure App Service -- PaaS
AWS S3 (Simple Storage Service) - IaaS
GitHub Codespaces - SaaS
Snowflake - PaaS
Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.

IaaS - Infrastructure as a Service requires more human intervention, but it comes blank. Services such as virtual machines, we can create all kinds that we need, but we have to provision and maintain it. 
PaaS - Platform as a Service provides platforms to provision our applications, while application code is brought and managed by us.
SaaS - Software as a Service are ready tools for us to use. That we don't have to worry at all about its backend, it is already complete and we just need to use it for our needs.

Cloud Concepts Question 4

Q: What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like Azure directly? What do you gain, and what do you give up?

A: Databricks or Snowflake stand above the cloud providers as underneath they use these cloud providers. What we gain is already managed services for us, we don't have to worry about setup or configurations, but we may give up in tems of privacy or just financial cost.

Cloud Concepts Question 5
Q: The lesson names two situations where the cloud is probably not the right choice. What are they?

A: 1) If dataset fits comfortably on a single machine and 2) no need for massive compute demands, local processing is often faster and cheaper.

Azure Basics

Azure Basics Question 1
What is the difference between an Azure subscription and a resource group? Which one is yours alone, and which one does CTD share?

A: Subscription is like an account and resource group is like a project inside that account. Mine is the resource group and CTD's is subsctiption.

Azure Basics Question 2
Azure Cloud Shell is ephemeral by default. What does that mean in practice, and what does your course setup use to make it persistent?

A: Ephemeral shell means whatever work we do there after closing we lose our work. To make it persisent, we need to mount a persistent storage that our work will be saved to. 

Azure Basics Question 3
What is the difference between your SSH private key and your SSH public key? Which one gets uploaded to the remote systems you want to connect to, and why is that safe?

A: Your private key stays secret on your machine and is used to prove your identity, while your public key is mathematically derived from it and can only verify — not recreate — the private key. The public key is uploaded to remote systems (e.g., ~/.ssh/), which is safe because even if someone obtains it, they cannot reverse-engineer your private key or impersonate you without it.

Azure Basics Question 4
Run the following command in Cloud Shell without the --output table flag:
az account show
Paste the output into your answer. Then describe in one sentence what changes when you add --output table.

A: 
{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#meerim.us11@gmail.com",
    "type": "user"
  }
}

It came out as an json output. Not that conveneint for readibility, so when we use that flag --output table, it outputs in a table on the shell, which is more human readible. 