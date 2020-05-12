from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core import Model
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.webservice import AciWebservice, Webservice, LocalWebservice

sp = ServicePrincipalAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47", # tenantID
                                    service_principal_id="2faad9e6-8bf9-4e7d-a732-70e3daa5ffd5", # clientId
                                    service_principal_password="##########") # clientSecret


# Create an environment and add conda dependencies to it
myenv = Environment(name="myenv")
# Enable Docker based environment
myenv.docker.enabled = True
# Build conda dependencies

myenv.python.conda_dependencies = CondaDependencies.create(conda_packages=['python==3.7.6','scikit-learn','tensorflow==2.1.0','pandas','numpy','matplotlib'],
                                                           pip_packages=['azureml-defaults','inference-schema[numpy-support]'])
inference_config = InferenceConfig(entry_script="score.py", environment=myenv, source_directory='packagefordeployment/')
ws = Workspace('bd04922c-a444-43dc-892f-74d5090f8a9a', 'mlplayarearg', 'testdeployment',auth=sp)

model = Model(workspace=ws, name='sentimentanalysisv2')

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 2, memory_gb = 6,auth_enabled=True)
#deployment_config = LocalWebservice.deploy_configuration(port=8890)

service = Model.deploy(ws, "positiveornegativev2", [model], inference_config, deployment_config, overwrite=True)
service.wait_for_deployment(show_output = True)
print(service.state)


print('dodododo')