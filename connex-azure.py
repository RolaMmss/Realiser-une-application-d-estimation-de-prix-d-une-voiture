from azureml.core import Workspace, Experiment, Environment
from azureml.train.automl import AutoMLConfig
from azureml.core import ScriptRunConfig


# ws = Workspace.from_config()

# Connect to your Azure ML workspace
ws = Workspace.get(name='rola_prix-voiture',                
                   subscription_id='111aaa69-41b9-4dfd-b6af-2ada039dd1ae',
                   resource_group='RG_SADEK')

# Créez un objet computetarget qui represente votre ressource de calcul distante
compute_target = ws.compute_targets["carprice"]

# Create a objet environment pour specifier 
my_env = Environment.from_pip_requirements(name='my_env', file_path='requirements.txt')

# Créez une configuration de script pour spécifier l'environnement d'exécution et les dépendances de votre expérience. 
src = ScriptRunConfig(source_directory='.',
                                   script='train.py',
                                   compute_target=compute_target,
                                   environment=my_env)

experiment = Experiment(workspace=ws, name='experiment_price')
# Soumettez votre expérience à l'exécution en utilisant la méthode submit() de votre objet Experiment.
run = experiment.submit(config=src)

run.wait_for_completion(show_output=True)
