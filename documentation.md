# Text from https://ibm.github.io/watsonx-ai-python-sdk/pt_model_inference.html








Tuned Model Inference - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](#)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Tuned Model Inference[¶](#tuned-model-inference "Link to this heading")
=======================================================================


This section shows how to deploy model and use `ModelInference` class with created deployment.


There are two ways to query `generate_text` using the [deployments](#generate-text-deployments) module or using [ModelInference](#generate-text-modelinference) module .



Working with deployments[¶](#working-with-deployments "Link to this heading")
-----------------------------------------------------------------------------


This section describes methods that enable user to work with deployments. But first it will be needed to create client and set `project_id` or `space_id`.



```
from ibm_watsonx_ai import APIClient

client = APIClient(credentials)
client.set.default_project("7ac03029-8bdd-4d5f-a561-2c4fd1e40705")

```


To create deployment with specific parameters call following lines.



```
from datetime import datetime

model_id = prompt_tuner.get_model_id()

meta_props = {
    client.deployments.ConfigurationMetaNames.NAME: "PT DEPLOYMENT SDK - project",
    client.deployments.ConfigurationMetaNames.ONLINE: {},
    client.deployments.ConfigurationMetaNames.SERVING_NAME : f"pt_sdk_deployment_{datetime.utcnow().strftime('%Y_%m_%d_%H%M%S')}"
}
deployment_details = client.deployments.create(model_id, meta_props)

```


To get a deployment\_id from details, use `id` from `metadata`.



```
deployment_id = deployment_details['metadata']['id']
print(deployment_id)
'7091629c-f88a-4e90-b7f0-4f414aec9c3a'

```


You can directly query `generate_text` using the deployments module.



```
client.deployments.generate_text(
    prompt="Example prompt",
    deployment_id=deployment_id)

```




Creating `ModelInference` instance[¶](#creating-modelinference-instance "Link to this heading")
-----------------------------------------------------------------------------------------------


At the beginning, it is recommended to define parameters (later used by module).



```
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

generate_params = {
    GenParams.MAX_NEW_TOKENS: 25,
    GenParams.STOP_SEQUENCES: ["\n"]
}

```


Create the ModelInference itself, using credentials and `project_id` / `space_id` or the previously initialized APIClient (see [APIClient initialization](#api-client-init)).



```
from ibm_watsonx_ai.foundation_models import ModelInference

tuned_model = ModelInference(
    deployment_id=deployment_id,
    params=generate_params,
    credentials=credentials,
    project_id=project_id
)

# OR

tuned_model = ModelInference(
    deployment_id=deployment_id,
    params=generate_params,
    api_client=client
)

```


You can directly query `generate_text` using the `ModelInference` object.



```
tuned_model.generate_text(prompt="Example prompt")

```




Importing data[¶](#importing-data "Link to this heading")
---------------------------------------------------------


To use ModelInference, an example data may be need.



```
import pandas as pd

filename = 'car_rental_prompt_tuning_testing_data.json'

url = "https://raw.github.com/IBM/watson-machine-learning-samples/master/cloud/data/prompt_tuning/car_rental_prompt_tuning_testing_data.json"
if not os.path.isfile(filename):
    wget.download(url)

data = pd.read_json(filename)

```




Analyzing satisfaction[¶](#analyzing-satisfaction "Link to this heading")
-------------------------------------------------------------------------



Note


The satisfaction analysis was performed for a specific example - **car rental**, it may not work in the case of other data sets.



To analyze satisfaction prepare batch with prompts, calculate the accuracy of tuned model and compare it with base model.



```
prompts = list(data.input)
satisfaction = list(data.output)
prompts_batch = ["\n".join([prompt]) for prompt in prompts]

```


Calculate accuracy of based model:



```
from sklearn.metrics import accuracy_score, f1_score

base_model = ModelInference(
    model_id='google/flan-t5-xl',
    params=generate_params,
    api_client=client
)
base_model_results = base_model.generate_text(prompt=prompts_batch)
print(f'base model accuracy_score: {accuracy_score(satisfaction, [int(x) for x in base_model_results])}, base model f1_score: {f1_score(satisfaction, [int(x) for x in base_model_results])}')
'base model accuracy_score: 0.965034965034965, base model f1_score: 0.9765258215962441'

```


Calculate accuracy of tuned model:



```
tuned_model_results = tuned_model.generate_text(prompt=prompts_batch)
print(f'accuracy_score: {accuracy_score(satisfaction, [int(x) for x in tuned_model_results])}, f1_score: {f1_score(satisfaction, [int(x) for x in tuned_model_results])}')
'accuracy_score: 0.972027972027972, f1_score: 0.9811320754716981'

```




Generate methods[¶](#generate-methods "Link to this heading")
-------------------------------------------------------------


The detailed explanation of available generate methods with exact parameters can be found in the [ModelInferece class](fm_model_inference.html#model-inference-class).


With previously created `tuned_model` object, it is possible to generate a text stream (generator) using defined inference and `generate_text_stream()` method.



```
for token in tuned_model.generate_text_stream(prompt=input_prompt):
    print(token, end="")
'$10 Powerchill Leggings'

```


And also receive more detailed result with `generate()`.



```
details = tuned_model.generate(prompt=input_prompt, params=gen_params)
print(details)
{
    'model_id': 'google/flan-t5-xl',
    'created_at': '2023-11-17T15:32:57.401Z',
    'results': [
        {
        'generated_text': '$10 Powerchill Leggings',
        'generated_token_count': 8,
        'input_token_count': 73,
        'stop_reason': 'eos_token'
        }
    ],
    'system': {'warnings': []}
}

```








[Next

Tune Experiment](tune_experiment.html)
[Previous

Tune Experiment run](pt_tune_experiment_run.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Tuned Model Inference](#)
	+ [Working with deployments](#working-with-deployments)
	+ [Creating `ModelInference` instance](#creating-modelinference-instance)
	+ [Importing data](#importing-data)
	+ [Analyzing satisfaction](#analyzing-satisfaction)
	+ [Generate methods](#generate-methods)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/fm_model_inference.html








ModelInference - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](#)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





ModelInference[¶](#modelinference "Link to this heading")
=========================================================




*class* ibm\_watsonx\_ai.foundation\_models.inference.ModelInference(*\**, *model\_id=None*, *deployment\_id=None*, *params=None*, *credentials=None*, *project\_id=None*, *space\_id=None*, *verify=None*, *api\_client=None*, *validate=True*)[[source]](_modules/ibm_watsonx_ai/foundation_models/inference/model_inference.html#ModelInference)[¶](#ibm_watsonx_ai.foundation_models.inference.ModelInference "Link to this definition")
Bases: `BaseModelInference`


Instantiate the model interface.



Hint


To use the ModelInference class with LangChain, use the `WatsonxLLM` wrapper.




Parameters:
* **model\_id** (*str**,* *optional*) – the type of model to use
* **deployment\_id** (*str**,* *optional*) – ID of tuned model’s deployment
* **credentials** (*ibm\_watsonx\_ai.Credentials* *or* *dict**,* *optional*) – credentials to Watson Machine Learning instance
* **params** (*dict**,* *optional*) – parameters to use during generate requests
* **project\_id** (*str**,* *optional*) – ID of the Watson Studio project
* **space\_id** (*str**,* *optional*) – ID of the Watson Studio space
* **verify** (*bool* *or* *str**,* *optional*) – user can pass as verify one of following:



	+ the path to a CA\_BUNDLE file
	+ the path of directory with certificates of trusted CAs
	+ True - default path to truststore will be taken
	+ False - no verification will be made
* **api\_client** ([*APIClient*](base.html#client.APIClient "client.APIClient")*,* *optional*) – Initialized APIClient object with set project or space ID. If passed, `credentials` and `project_id`/`space_id` are not required.
* **validate** (*bool**,* *optional*) – Model id validation, defaults to True





Note


One of these parameters is required: [`model_id`, `deployment_id`]




Note


One of these parameters is required: [`project_id`, `space_id`] when `credentials` parameter passed.




Hint


You can copy the project\_id from the Project’s Manage tab (Project -> Manage -> General -> Details).



**Example**



```
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods

# To display example params enter
GenParams().get_example_values()

generate_params = {
    GenParams.MAX_NEW_TOKENS: 25
}

model_inference = ModelInference(
    model_id=ModelTypes.FLAN_UL2,
    params=generate_params,
    credentials=Credentials(
                        api_key = "***",
                        url = "https://us-south.ml.cloud.ibm.com"),
    project_id="*****"
    )

```



```
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials

deployment_inference = ModelInference(
    deployment_id="<ID of deployed model>",
    credentials=Credentials(
                    api_key = "***",
                    url = "https://us-south.ml.cloud.ibm.com"),
    project_id="*****"
    )

```




generate(*prompt=None*, *params=None*, *guardrails=False*, *guardrails\_hap\_params=None*, *guardrails\_pii\_params=None*, *concurrency\_limit=10*, *async\_mode=False*, *validate\_prompt\_variables=True*)[[source]](_modules/ibm_watsonx_ai/foundation_models/inference/model_inference.html#ModelInference.generate)[¶](#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate "Link to this definition")
Given a text prompt as input, and parameters the selected model (model\_id) or deployment (deployment\_id)
will generate a completion text as generated\_text. For prompt template deployment prompt should be None.



Parameters:
* **params** (*dict*) – meta props for text generation, use `ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()` to view the list of MetaNames
* **concurrency\_limit** (*int*) – number of requests that will be sent in parallel, max is 10
* **prompt** (*(**str* *|* *list* *|* *None**)**,* *optional*) – the prompt string or list of strings. If list of strings is passed requests will be managed in parallel with the rate of concurency\_limit, defaults to None
* **guardrails** (*bool*) – If True then potentially hateful, abusive, and/or profane language (HAP) detection
filter is toggle on for both prompt and generated text, defaults to False
* **guardrails\_hap\_params** (*dict*) – meta props for HAP moderations, use `ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()`
to view the list of MetaNames
* **async\_mode** (*bool*) – If True then yield results asynchronously (using generator). In this case both prompt and
generated text will be concatenated in the final response - under generated\_text, defaults
to False
* **validate\_prompt\_variables** (*bool*) – If True, prompt variables provided in params are validated with the ones in Prompt Template Asset.
This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True



Returns:
scoring result containing generated content



Return type:
dict




**Example**



```
q = "What is 1 + 1?"
generated_response = model_inference.generate(prompt=q)
print(generated_response['results'][0]['generated_text'])

```





generate\_text(*prompt=None*, *params=None*, *raw\_response=False*, *guardrails=False*, *guardrails\_hap\_params=None*, *guardrails\_pii\_params=None*, *concurrency\_limit=10*, *validate\_prompt\_variables=True*)[[source]](_modules/ibm_watsonx_ai/foundation_models/inference/model_inference.html#ModelInference.generate_text)[¶](#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate_text "Link to this definition")
Given a text prompt as input, and parameters the selected model (model\_id)
will generate a completion text as generated\_text. For prompt template deployment prompt should be None.



Parameters:
* **params** (*dict*) – meta props for text generation, use `ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()` to view the list of MetaNames
* **concurrency\_limit** (*int*) – number of requests that will be sent in parallel, max is 10
* **prompt** (*(**str* *|* *list* *|* *None**)**,* *optional*) – the prompt string or list of strings. If list of strings is passed requests will be managed in parallel with the rate of concurency\_limit, defaults to None
* **guardrails** (*bool*) – If True then potentially hateful, abusive, and/or profane language (HAP) detection filter is toggle on for both prompt and generated text, defaults to False
If HAP is detected the HAPDetectionWarning is issued
* **guardrails\_hap\_params** (*dict*) – meta props for HAP moderations, use `ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()`
to view the list of MetaNames
* **raw\_response** (*bool**,* *optional*) – return the whole response object
* **validate\_prompt\_variables** (*bool*) – If True, prompt variables provided in params are validated with the ones in Prompt Template Asset.
This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True



Returns:
generated content



Return type:
str | list | dict





Note


By default only the first occurrence of HAPDetectionWarning is displayed. To enable printing all warnings of this category, use:



```
import warnings
from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

warnings.filterwarnings("always", category=HAPDetectionWarning)

```



**Example**



```
q = "What is 1 + 1?"
generated_text = model_inference.generate_text(prompt=q)
print(generated_text)

```





generate\_text\_stream(*prompt=None*, *params=None*, *raw\_response=False*, *guardrails=False*, *guardrails\_hap\_params=None*, *guardrails\_pii\_params=None*, *validate\_prompt\_variables=True*)[[source]](_modules/ibm_watsonx_ai/foundation_models/inference/model_inference.html#ModelInference.generate_text_stream)[¶](#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate_text_stream "Link to this definition")
Given a text prompt as input, and parameters the selected model (model\_id)
will generate a streamed text as generate\_text\_stream. For prompt template deployment prompt should be None.



Parameters:
* **params** (*dict*) – meta props for text generation, use `ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()` to view the list of MetaNames
* **prompt** (*str**,* *optional*) – the prompt string, defaults to None
* **raw\_response** (*bool**,* *optional*) – yields the whole response object
* **guardrails** (*bool*) – If True then potentially hateful, abusive, and/or profane language (HAP) detection filter is toggle on for both prompt and generated text, defaults to False
If HAP is detected the HAPDetectionWarning is issued
* **guardrails\_hap\_params** (*dict*) – meta props for HAP moderations, use `ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()`
to view the list of MetaNames
* **validate\_prompt\_variables** (*bool*) – If True, prompt variables provided in params are validated with the ones in Prompt Template Asset.
This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True



Returns:
scoring result containing generated content



Return type:
generator





Note


By default only the first occurrence of HAPDetectionWarning is displayed. To enable printing all warnings of this category, use:



```
import warnings
from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

warnings.filterwarnings("always", category=HAPDetectionWarning)

```



**Example**



```
q = "Write an epigram about the sun"
generated_response = model_inference.generate_text_stream(prompt=q)

for chunk in generated_response:
    print(chunk, end='', flush=True)

```





get\_details()[[source]](_modules/ibm_watsonx_ai/foundation_models/inference/model_inference.html#ModelInference.get_details)[¶](#ibm_watsonx_ai.foundation_models.inference.ModelInference.get_details "Link to this definition")
Get model interface’s details



Returns:
details of model or deployment



Return type:
dict




**Example**



```
model_inference.get_details()

```





get\_identifying\_params()[[source]](_modules/ibm_watsonx_ai/foundation_models/inference/model_inference.html#ModelInference.get_identifying_params)[¶](#ibm_watsonx_ai.foundation_models.inference.ModelInference.get_identifying_params "Link to this definition")
Represent Model Inference’s setup in dictionary





to\_langchain()[[source]](_modules/ibm_watsonx_ai/foundation_models/inference/model_inference.html#ModelInference.to_langchain)[¶](#ibm_watsonx_ai.foundation_models.inference.ModelInference.to_langchain "Link to this definition")

Returns:
WatsonxLLM wrapper for watsonx foundation models



Return type:
[WatsonxLLM](fm_extensions.html#langchain_ibm.WatsonxLLM "langchain_ibm.WatsonxLLM")




**Example**



```
from langchain import PromptTemplate
from langchain.chains import LLMChain
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

flan_ul2_model = ModelInference(
    model_id=ModelTypes.FLAN_UL2,
    credentials=Credentials(
                        api_key = "***",
                        url = "https://us-south.ml.cloud.ibm.com"),
    project_id="*****"
    )

prompt_template = "What color is the {flower}?"

llm_chain = LLMChain(llm=flan_ul2_model.to_langchain(), prompt=PromptTemplate.from_template(prompt_template))
llm_chain('sunflower')

```



```
from langchain import PromptTemplate
from langchain.chains import LLMChain
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

deployed_model = ModelInference(
    deployment_id="<ID of deployed model>",
    credentials=Credentials(
                        api_key = "***",
                        url = "https://us-south.ml.cloud.ibm.com"),
    space_id="*****"
    )

prompt_template = "What color is the {car}?"

llm_chain = LLMChain(llm=deployed_model.to_langchain(), prompt=PromptTemplate.from_template(prompt_template))
llm_chain('sunflower')

```





tokenize(*prompt*, *return\_tokens=False*)[[source]](_modules/ibm_watsonx_ai/foundation_models/inference/model_inference.html#ModelInference.tokenize)[¶](#ibm_watsonx_ai.foundation_models.inference.ModelInference.tokenize "Link to this definition")
The text tokenize operation allows you to check the conversion of provided input to tokens for a given model.
It splits text into words or sub-words, which then are converted to ids through a look-up table (vocabulary).
Tokenization allows the model to have a reasonable vocabulary size.



Note


Method is not supported for deployments, available only for base models.




Parameters:
* **prompt** (*str**,* *optional*) – the prompt string, defaults to None
* **return\_tokens** (*bool*) – the parameter for text tokenization, defaults to False



Returns:
the result of tokenizing the input string.



Return type:
dict




**Example**



```
q = "Write an epigram about the moon"
tokenized_response = model_inference.tokenize(prompt=q, return_tokens=True)
print(tokenized_response["result"])

```









[Next

`ModelInference` for Deployments](fm_deployments.html)
[Previous

Model](fm_model.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [ModelInference](#)
	+ [`ModelInference`](#ibm_watsonx_ai.foundation_models.inference.ModelInference)
		- [`ModelInference.generate()`](#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate)
		- [`ModelInference.generate_text()`](#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate_text)
		- [`ModelInference.generate_text_stream()`](#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate_text_stream)
		- [`ModelInference.get_details()`](#ibm_watsonx_ai.foundation_models.inference.ModelInference.get_details)
		- [`ModelInference.get_identifying_params()`](#ibm_watsonx_ai.foundation_models.inference.ModelInference.get_identifying_params)
		- [`ModelInference.to_langchain()`](#ibm_watsonx_ai.foundation_models.inference.ModelInference.to_langchain)
		- [`ModelInference.tokenize()`](#ibm_watsonx_ai.foundation_models.inference.ModelInference.tokenize)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/index.html








IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](#)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](#)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





The ibm-watsonx-ai library[¶](#the-ibm-watsonx-ai-library "Link to this heading")
=================================================================================


The `ibm-watsonx-ai` Python library allows you to work with IBM watsonx.ai services.
You can train, store, and deploy your models, score them using APIs, and finally integrate them with your application
development.


For supported product offerings refer to [Product Offerings](install.html#product-offerings) section.



* [Installation](install.html)
	+ [Product Offerings](install.html#product-offerings)
* [Setup](setup.html)
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
		- [Requirements](setup_cloud.html#requirements)
		- [Supported machine learning frameworks](setup_cloud.html#supported-machine-learning-frameworks)
		- [Authentication](setup_cloud.html#authentication)
		- [Firewall settings](setup_cloud.html#firewall-settings)
	+ [IBM watsonx.ai software](setup_cpd.html)
		- [Requirements](setup_cpd.html#requirements)
		- [Supported machine learning frameworks](setup_cpd.html#supported-machine-learning-frameworks)
		- [Authentication](setup_cpd.html#authentication)
* [API](api.html)
	+ [Prerequisites](api.html#prerequisites)
	+ [Modules](api.html#modules)
		- [Base](base.html)
		- [Core](core_api.html)
		- [Federated Learning](federated_learning.html)
		- [DataConnection](dataconnection.html)
		- [AutoAI](autoai.html)
		- [Foundation Models](foundation_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
	+ [Migration Steps to New Packages](migration.html#migration-steps-to-new-packages)
* [V1 Migration Guide](migration_v1.html)
	+ [What’s new?](migration_v1.html#what-s-new)
		- [Refactor and Cleanup](migration_v1.html#refactor-and-cleanup)
		- [New authorization](migration_v1.html#new-authorization)
		- [Easier project and space setup](migration_v1.html#easier-project-and-space-setup)
		- [Foundation models functions moved under APIClient](migration_v1.html#foundation-models-functions-moved-under-apiclient)
		- [Breaking changes](migration_v1.html#breaking-changes)
		- [Deprecations](migration_v1.html#deprecations)
* [Changelog](changelog.html)
	+ [1.0.10](changelog.html#id1)
	+ [1.0.9](changelog.html#id2)
	+ [1.0.8](changelog.html#id3)
	+ [1.0.6](changelog.html#id4)
	+ [1.0.5](changelog.html#id5)
	+ [1.0.4](changelog.html#id6)
	+ [1.0.3](changelog.html#id7)
	+ [1.0.2](changelog.html#id8)
	+ [1.0.1](changelog.html#id9)
	+ [1.0.0](changelog.html#id10)





Indices and tables[¶](#indices-and-tables "Link to this heading")
=================================================================


* [Index](genindex.html)
* [Module Index](py-modindex.html)
* [Search Page](search.html)







[Next

Installation](install.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)
















# Text from https://ibm.github.io/watsonx-ai-python-sdk/migration.html








Migration from ibm\_watson\_machine\_learning - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](#)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Migration from `ibm_watson_machine_learning`[¶](#migration-from-ibm-watson-machine-learning "Link to this heading")
===================================================================================================================


New `ibm_watsonx_ai` Python SDK is upgraded version of `ibm_watson_machine_learning`.
They are similar in usage, however new Python SDK is extended with new functionality and cleared of outdated code.
Your first attempt to migrate should be update the imports to new Python SDK (as seen below) and see if this is sufficient change by running the code.
Observe if any warnings or errors are raised, which would indicate need of further changes.



Migration Steps to New Packages[¶](#migration-steps-to-new-packages "Link to this heading")
-------------------------------------------------------------------------------------------


1. **Installation and import**:


To install `ibm_watsonx_ai`, run:



```
$ pip install -U ibm-watsonx-ai

```


To import components from `ibm_watsonx_ai`, (example of importing `APIClient`) run:



```
from ibm_watsonx_ai import APIClient

```








[Next

V1 Migration Guide](migration_v1.html)
[Previous

Samples](samples.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Migration from `ibm_watson_machine_learning`](#)
	+ [Migration Steps to New Packages](#migration-steps-to-new-packages)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/federated_learning.html








Federated Learning - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](#)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Federated Learning[¶](#federated-learning "Link to this heading")
=================================================================


Federated Learning provides the tools for training a model collaboratively, by coordinating local training runs
and fusing the results. Even though data sources are never moved, combined, or shared among parties or the aggregator,
all of them contribute to training and improving the quality of the global model.


[Tutorial and Samples for IBM watsonx.ai for IBM Cloud](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fl-demo.html?context=wx&audience=wdp)


[Tutorial and Samples for IBM watsonx.ai software, IBM watsonx.ai Server](https://www.ibm.com/docs/en/cloud-paks/cp-data/4.0?topic=learning-federated-tutorial-samples)



Aggregation[¶](#aggregation "Link to this heading")
---------------------------------------------------


The aggregator process, which fuses the parties’ training results, runs as a watsonx.ai training job.
For more information on creating and querying a training job, see the API documentation for the `client.training` class.
The parameters available to configure a Federated Learning training are described in the [IBM Cloud API Docs](https://cloud.ibm.com/apidocs/machine-learning-cp#trainings-create).



### Configure and start aggregation[¶](#configure-and-start-aggregation "Link to this heading")



```
from ibm_watsonx_ai import APIClient

client = APIClient( credentials )

PROJECT_ID = "8ae1a720-83ed-4c57-b719-8dd086bd7ce0"
client.set.default_project( PROJECT_ID )

aggregator_metadata = {
    client.training.ConfigurationMetaNames.NAME: 'Federated Tensorflow MNIST',
    client.training.ConfigurationMetaNames.DESCRIPTION: 'MNIST digit recognition with Federated Learning using Tensorflow',
    client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [],
    client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
        'type': 'container',
        'name': 'outputData',
        'connection': {},
        'location': {
           'path': '/projects/' + PROJECT_ID + '/assets/trainings/'
        }

    },
    client.training.ConfigurationMetaNames.FEDERATED_LEARNING: {
        'model': {
            'type': 'tensorflow',
            'spec': {
               'id': untrained_model_id
            },
            'model_file': untrained_model_name
        },
        'fusion_type': 'iter_avg',
        'metrics': 'accuracy',
        'epochs': 3,
        'rounds': 99,
        'remote_training' : {
            'quorum': 1.0,
            'max_timeout': 3600,
            'remote_training_systems': [ { 'id': rts_1_id }, { 'id': rts_2_id} ]
        },
        'hardware_spec': {
            'name': 'S'
        },
        'software_spec': {
            'name': 'runtime-22.1-py3.9'
        }
    }
}


aggregator = client.training.run(aggregator_metadata, asynchronous=True)
aggregator_id = client.training.get_id(aggregator)

```





Local training[¶](#local-training "Link to this heading")
---------------------------------------------------------


Training is performed locally by parties which connect to the aggregator. The parties must be members of the project
or space in which the aggregator is running and are identified to the Federated Learning aggregator as Remote Training Systems.



### Configure and start local training[¶](#configure-and-start-local-training "Link to this heading")



```
from ibm_watsonx_ai import APIClient

client = APIClient( party_1_credentials )

PROJECT_ID = "8ae1a720-83ed-4c57-b719-8dd086bd7ce0"
client.set.default_project( PROJECT_ID )

# The party needs, at mimimum, to specify how the data are loaded for training.  The data
# handler class and any input to the class is provided.  In this case, the info block
# contains a key to locate the training data from the current working directory.
party_metadata = {
                    client.remote_training_systems.ConfigurationMetaNames.DATA_HANDLER: {
                       "class": MNISTDataHandler,
                       "info": {
                          "npz_file": "./training_data.npz"
                       }
                 }
# The party object is created
party = client.remote_training_systems.create_party(remote_training_system_id = "d516d42c-6c59-41f2-b7ca-c63d11ea79a1", party_metadata)
# Send training logging to standard output
party.monitor_logs()
# Start training.  Training will run in the Python process that is executing this code.
# The supplied aggregator_id refers to the watsonx.ai training job that will perform aggregation.
party.run(aggregator_id = "564fb126-9bfd-409b-beb3-5d401e4c50ec", asynchronous = False)

```




*class* remote\_training\_system.RemoteTrainingSystem(*client*)[[source]](_modules/remote_training_system.html#RemoteTrainingSystem)[¶](#remote_training_system.RemoteTrainingSystem "Link to this definition")
The RemoteTrainingSystem class represents a Federated Learning party and provides a list of identities
that are permitted to join training as the RemoteTrainingSystem.




create\_party(*remote\_training\_system\_id*, *party\_metadata*)[[source]](_modules/remote_training_system.html#RemoteTrainingSystem.create_party)[¶](#remote_training_system.RemoteTrainingSystem.create_party "Link to this definition")
Create a party object using the specified remote training system id and the party metadata.



Parameters:
* **remote\_training\_system\_id** (*str*) – remote training system identifier
* **party\_metadata** (*dict*) – the party configuration



Returns:
a party object with the specified rts\_id and configuration



Return type:
[Party](#party_wrapper.Party "party_wrapper.Party")




**Examples**



```
party_metadata = {
    client.remote_training_systems.ConfigurationMetaNames.DATA_HANDLER: {
        "info": {
            "npz_file": "./data_party0.npz"
        },
        "name": "MnistTFDataHandler",
        "path": "./mnist_keras_data_handler.py"
    },
    client.remote_training_systems.ConfigurationMetaNames.LOCAL_TRAINING: {
        "name": "LocalTrainingHandler",
        "path": "ibmfl.party.training.local_training_handler"
    },
    client.remote_training_systems.ConfigurationMetaNames.HYPERPARAMS: {
        "epochs": 3
    },
}
party = client.remote_training_systems.create_party(remote_training_system_id, party_metadata)

```



```
party_metadata = {
    client.remote_training_systems.ConfigurationMetaNames.DATA_HANDLER: {
        "info": {
            "npz_file": "./data_party0.npz"
        },
        "class": MnistTFDataHandler
    }
}
party = client.remote_training_systems.create_party(remote_training_system_id, party_metadata)

```





create\_revision(*remote\_training\_system\_id*)[[source]](_modules/remote_training_system.html#RemoteTrainingSystem.create_revision)[¶](#remote_training_system.RemoteTrainingSystem.create_revision "Link to this definition")
Create a new remote training system revision.



Parameters:
**remote\_training\_system\_id** (*str*) – Unique remote training system ID



Returns:
remote training system details



Return type:
dict




**Example**



```
client.remote_training_systems.create_revision(remote_training_system_id)

```





delete(*remote\_training\_systems\_id*)[[source]](_modules/remote_training_system.html#RemoteTrainingSystem.delete)[¶](#remote_training_system.RemoteTrainingSystem.delete "Link to this definition")
Deletes the given remote\_training\_systems\_id definition. space\_id or project\_id has to be provided.



Parameters:
**remote\_training\_systems\_id** (*str*) – remote training system identifier



Returns:
status (“SUCCESS” or “FAILED”)



Return type:
str




**Example**



```
client.remote_training_systems.delete(remote_training_systems_id='6213cf1-252f-424b-b52d-5cdd9814956c')

```





get\_details(*remote\_training\_system\_id=None*, *limit=None*, *asynchronous=False*, *get\_all=False*)[[source]](_modules/remote_training_system.html#RemoteTrainingSystem.get_details)[¶](#remote_training_system.RemoteTrainingSystem.get_details "Link to this definition")

Get metadata of the given remote training system. If remote\_training\_system\_id is not specified,metadata is returned for all remote training systems.





Parameters:
* **remote\_training\_system\_id** (*str**,* *optional*) – remote training system identifier
* **limit** (*int**,* *optional*) – limit number of fetched records
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks



Returns:
remote training system(s) metadata



Return type:
dict (if remote\_training\_systems\_id is not None) or {“resources”: [dict]} (if remote\_training\_systems\_id is None)




**Examples**



```
details = client.remote_training_systems.get_details(remote_training_systems_id)
details = client.remote_training_systems.get_details()
details = client.remote_training_systems.get_details(limit=100)
details = client.remote_training_systems.get_details(limit=100, get_all=True)
details = []
for entry in client.remote_training_systems.get_details(limit=100, asynchronous=True, get_all=True):
    details.extend(entry)

```





*static* get\_id(*remote\_training\_system\_details*)[[source]](_modules/remote_training_system.html#RemoteTrainingSystem.get_id)[¶](#remote_training_system.RemoteTrainingSystem.get_id "Link to this definition")
Get ID of remote training system.



Parameters:
**remote\_training\_system\_details** (*dict*) – metadata of the stored remote training system



Returns:
ID of stored remote training system



Return type:
str




**Example**



```
details = client.remote_training_systems.get_details(remote_training_system_id)
id = client.remote_training_systems.get_id(details)

```





get\_revision\_details(*remote\_training\_system\_id*, *rev\_id*)[[source]](_modules/remote_training_system.html#RemoteTrainingSystem.get_revision_details)[¶](#remote_training_system.RemoteTrainingSystem.get_revision_details "Link to this definition")
Get metadata from the specific revision of a stored remote system.



Parameters:
* **remote\_training\_system\_id** (*str*) – ID of remote training system
* **rev\_id** (*str*) – Unique id of the remote system revision



Returns:
stored remote system revision metadata



Return type:
dict




Example:



```
details = client.remote_training_systems.get_details(remote_training_system_id, rev_id)

```





list(*limit=None*)[[source]](_modules/remote_training_system.html#RemoteTrainingSystem.list)[¶](#remote_training_system.RemoteTrainingSystem.list "Link to this definition")
Lists stored remote training systems in a table format.
If limit is set to None, only the first 50 records are shown.



Parameters:
**limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed remote training systems



Return type:
pandas.DataFrame




**Example**



```
client.remote_training_systems.list()

```





list\_revisions(*remote\_training\_system\_id*, *limit=None*)[[source]](_modules/remote_training_system.html#RemoteTrainingSystem.list_revisions)[¶](#remote_training_system.RemoteTrainingSystem.list_revisions "Link to this definition")
Print all revisions for the given remote\_training\_system\_id in a table format.



Parameters:
* **remote\_training\_system\_id** (*str*) – Unique id of stored remote system
* **limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed remote training system revisions



Return type:
pandas.DataFrame




**Example**



```
client.remote_training_systems.list_revisions(remote_training_system_id)

```





store(*meta\_props*)[[source]](_modules/remote_training_system.html#RemoteTrainingSystem.store)[¶](#remote_training_system.RemoteTrainingSystem.store "Link to this definition")
Create a remote training system. Either space\_id or project\_id has to be provided.



Parameters:
**meta\_props** (*dict*) – metadata, to see available meta names use
`client.remote_training_systems.ConfigurationMetaNames.get()`



Returns:
response json



Return type:
dict




**Example**



```
metadata = {
    client.remote_training_systems.ConfigurationMetaNames.NAME: "my-resource",
    client.remote_training_systems.ConfigurationMetaNames.TAGS: ["tag1", "tag2"],
    client.remote_training_systems.ConfigurationMetaNames.ORGANIZATION: {"name": "name", "region": "EU"}
    client.remote_training_systems.ConfigurationMetaNames.ALLOWED_IDENTITIES: [{"id": "43689024", "type": "user"}],
    client.remote_training_systems.ConfigurationMetaNames.REMOTE_ADMIN: {"id": "43689020", "type": "user"}
}
client.set.default_space('3fc54cf1-252f-424b-b52d-5cdd9814987f')
details = client.remote_training_systems.store(meta_props=metadata)

```





update(*remote\_training\_system\_id*, *changes*)[[source]](_modules/remote_training_system.html#RemoteTrainingSystem.update)[¶](#remote_training_system.RemoteTrainingSystem.update "Link to this definition")
Updates existing remote training system metadata.



Parameters:
* **remote\_training\_system\_id** (*str*) – remote training system identifier
* **changes** (*dict*) – elements which should be changed, where keys are ConfigurationMetaNames



Returns:
updated remote training system details



Return type:
dict




**Example**



```
metadata = {
    client.remote_training_systems.ConfigurationMetaNames.NAME:"updated_remote_training_system"
}
details = client.remote_training_systems.update(remote_training_system_id, changes=metadata)

```






*class* party\_wrapper.Party(*client=None*, *\*\*kwargs*)[[source]](_modules/party_wrapper.html#Party)[¶](#party_wrapper.Party "Link to this definition")
The Party class embodies a Federated Learning party, with methods to run, cancel, and query local training.
Refer to the `client.remote_training_system.create_party()` API for more information about creating an
instance of the Party class.




cancel()[[source]](_modules/party_wrapper.html#Party.cancel)[¶](#party_wrapper.Party.cancel "Link to this definition")
Stop the local connection to the training on the party side.


**Example**



```
party.cancel()

```





get\_round()[[source]](_modules/party_wrapper.html#Party.get_round)[¶](#party_wrapper.Party.get_round "Link to this definition")
Get the current round number.



Returns:
the current round number



Return type:
int




**Example**



```
party.get_round()

```





is\_running()[[source]](_modules/party_wrapper.html#Party.is_running)[¶](#party_wrapper.Party.is_running "Link to this definition")
Check if the training job is running.



Returns:
if the job is running



Return type:
bool




**Example**



```
party.is_running()

```





monitor\_logs(*log\_level='INFO'*)[[source]](_modules/party_wrapper.html#Party.monitor_logs)[¶](#party_wrapper.Party.monitor_logs "Link to this definition")
Enable logging of the training job to standard output.
This method should be called before calling the `run()` method.



Parameters:
**log\_level** (*str**,* *optional*) – log level specified by user




**Example**



```
party.monitor_logs()

```





monitor\_metrics(*metrics\_file='-'*)[[source]](_modules/party_wrapper.html#Party.monitor_metrics)[¶](#party_wrapper.Party.monitor_metrics "Link to this definition")
Enable output of training metrics.



Parameters:
**metrics\_file** (*str**,* *optional*) – a filename specified by user to which the metrics should be written





Note


This method outputs the metrics to stdout if a filename is not specified



**Example**



```
party.monitor_metrics()

```





run(*aggregator\_id=None*, *experiment\_id=None*, *asynchronous=True*, *verify=True*, *timeout=600*)[[source]](_modules/party_wrapper.html#Party.run)[¶](#party_wrapper.Party.run "Link to this definition")
Connect to a Federated Learning aggregator and run local training.
Exactly one of aggregator\_id and experiment\_id must be supplied.



Parameters:
* **aggregator\_id** (*str**,* *optional*) – aggregator identifier



	+ If aggregator\_id is supplied, the party will connect to the given aggregator.
* **experiment\_id** (*str**,* *optional*) – experiment identifier



	+ If experiment\_id is supplied, the party will connect to the most recently created aggregatorfor the experiment.
* **asynchronous** (*bool**,* *optional*) – 
	+ True - party starts to run the job in the background and progress can be checked later
	+ False - method will wait until training is complete and then print the job status
* **verify** (*bool**,* *optional*) – verify certificate
* **timeout** (*int**, or* *None for no timeout*) – timeout in seconds



	+ If the aggregator is not ready within timeout seconds from now, exit.




**Examples**



```
party.run( aggregator_id = "69500105-9fd2-4326-ad27-7231aeb37ac8", asynchronous = True, verify = True )
party.run( experiment_id = "2466fa06-1110-4169-a166-01959adec995", asynchronous = False )

```











[Next

DataConnection](dataconnection.html)
[Previous

Core](core_api.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Federated Learning](#)
	+ [Aggregation](#aggregation)
		- [Configure and start aggregation](#configure-and-start-aggregation)
	+ [Local training](#local-training)
		- [Configure and start local training](#configure-and-start-local-training)
			* [`RemoteTrainingSystem`](#remote_training_system.RemoteTrainingSystem)
				+ [`RemoteTrainingSystem.create_party()`](#remote_training_system.RemoteTrainingSystem.create_party)
				+ [`RemoteTrainingSystem.create_revision()`](#remote_training_system.RemoteTrainingSystem.create_revision)
				+ [`RemoteTrainingSystem.delete()`](#remote_training_system.RemoteTrainingSystem.delete)
				+ [`RemoteTrainingSystem.get_details()`](#remote_training_system.RemoteTrainingSystem.get_details)
				+ [`RemoteTrainingSystem.get_id()`](#remote_training_system.RemoteTrainingSystem.get_id)
				+ [`RemoteTrainingSystem.get_revision_details()`](#remote_training_system.RemoteTrainingSystem.get_revision_details)
				+ [`RemoteTrainingSystem.list()`](#remote_training_system.RemoteTrainingSystem.list)
				+ [`RemoteTrainingSystem.list_revisions()`](#remote_training_system.RemoteTrainingSystem.list_revisions)
				+ [`RemoteTrainingSystem.store()`](#remote_training_system.RemoteTrainingSystem.store)
				+ [`RemoteTrainingSystem.update()`](#remote_training_system.RemoteTrainingSystem.update)
			* [`Party`](#party_wrapper.Party)
				+ [`Party.cancel()`](#party_wrapper.Party.cancel)
				+ [`Party.get_round()`](#party_wrapper.Party.get_round)
				+ [`Party.is_running()`](#party_wrapper.Party.is_running)
				+ [`Party.monitor_logs()`](#party_wrapper.Party.monitor_logs)
				+ [`Party.monitor_metrics()`](#party_wrapper.Party.monitor_metrics)
				+ [`Party.run()`](#party_wrapper.Party.run)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/fm_embeddings.html








Embeddings - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](#)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Embeddings[¶](#embeddings "Link to this heading")
=================================================



Embeddings[¶](#id1 "Link to this heading")
------------------------------------------




*class* ibm\_watsonx\_ai.foundation\_models.embeddings.Embeddings(*\**, *model\_id*, *params=None*, *credentials=None*, *project\_id=None*, *space\_id=None*, *api\_client=None*, *verify=None*)[[source]](_modules/ibm_watsonx_ai/foundation_models/embeddings/embeddings.html#Embeddings)[¶](#ibm_watsonx_ai.foundation_models.embeddings.Embeddings "Link to this definition")
Bases: [`BaseEmbeddings`](#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings "ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings"), `WMLResource`


Instantiate the embeddings service.



Parameters:
* **model\_id** (*str**,* *optional*) – the type of model to use
* **params** (*dict**,* *optional*) – parameters to use during generate requests, use `ibm_watsonx_ai.metanames.EmbedTextParamsMetaNames().show()` to view the list of MetaNames
* **credentials** (*dict**,* *optional*) – credentials to Watson Machine Learning instance
* **project\_id** (*str**,* *optional*) – ID of the Watson Studio project
* **space\_id** (*str**,* *optional*) – ID of the Watson Studio space
* **api\_client** ([*APIClient*](base.html#client.APIClient "client.APIClient")*,* *optional*) – Initialized APIClient object with set project or space ID. If passed, `credentials` and `project_id`/`space_id` are not required.
* **verify** (*bool* *or* *str**,* *optional*) – user can pass as verify one of following:



	+ the path to a CA\_BUNDLE file
	+ the path of directory with certificates of trusted CAs
	+ True - default path to truststore will be taken
	+ False - no verification will be made





Note


One of these parameters is required: [`project_id`, `space_id`] when `credentials` parameter passed.




Hint


You can copy the project\_id from the Project’s Manage tab (Project -> Manage -> General -> Details).



**Example**



```
 from ibm_watsonx_ai import Credentials
 from ibm_watsonx_ai.foundation_models import Embeddings
 from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
 from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

embed_params = {
     EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
     EmbedParams.RETURN_OPTIONS: {
     'input_text': True
     }
 }

 embedding = Embeddings(
     model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
     params=embed_params,
     credentials=Credentials(
         api_key = "***",
         url = "https://us-south.ml.cloud.ibm.com"),
     project_id="*****"
     )

```




embed\_documents(*texts*, *params=None*, *concurrency\_limit=10*)[[source]](_modules/ibm_watsonx_ai/foundation_models/embeddings/embeddings.html#Embeddings.embed_documents)[¶](#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.embed_documents "Link to this definition")
Return list of embedding vectors for provided texts.



Parameters:
* **texts** (*list**[**str**]*) – List of texts for which embedding vectors will be generated.
* **params** (*ParamsType* *|* *None**,* *optional*) – meta props for embedding generation, use `ibm_watsonx_ai.metanames.EmbedTextParamsMetaNames().show()` to view the list of MetaNames, defaults to None
* **concurrency\_limit** (*int**,* *optional*) – number of requests that will be sent in parallel, max is 10, defaults to 10



Returns:
List of embedding vectors



Return type:
list[list[float]]




**Example**



```
q = [
    "What is a Generative AI?",
    "Generative AI refers to a type of artificial intelligence that can original content."
    ]

embedding_vectors = embedding.embed_documents(texts=q)
print(embedding_vectors)

```





embed\_query(*text*, *params=None*)[[source]](_modules/ibm_watsonx_ai/foundation_models/embeddings/embeddings.html#Embeddings.embed_query)[¶](#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.embed_query "Link to this definition")
Return embedding vector for provided text.



Parameters:
* **text** (*str*) – Text for which embedding vector will be generated.
* **params** (*ParamsType* *|* *None**,* *optional*) – meta props for embedding generation, use `ibm_watsonx_ai.metanames.EmbedTextParamsMetaNames().show()` to view the list of MetaNames, defaults to None



Returns:
Embedding vector



Return type:
list[float]




**Example**



```
q = "What is a Generative AI?"
embedding_vector = embedding.embed_query(text=q)
print(embedding_vector)

```





generate(*inputs*, *params=None*, *concurrency\_limit=10*)[[source]](_modules/ibm_watsonx_ai/foundation_models/embeddings/embeddings.html#Embeddings.generate)[¶](#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.generate "Link to this definition")
Generates embeddings vectors for the given input with the given
parameters and returns a REST API response.



Parameters:
* **inputs** (*list**[**str**]*) – List of texts for which embedding vectors will be generated.
* **params** (*ParamsType* *|* *None**,* *optional*) – meta props for embedding generation, use `ibm_watsonx_ai.metanames.EmbedTextParamsMetaNames().show()` to view the list of MetaNames, defaults to None
* **concurrency\_limit** (*int**,* *optional*) – number of requests that will be sent in parallel, max is 10, defaults to 10



Returns:
scoring results containing generated embeddings vectors



Return type:
dict







to\_dict()[[source]](_modules/ibm_watsonx_ai/foundation_models/embeddings/embeddings.html#Embeddings.to_dict)[¶](#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.to_dict "Link to this definition")
Serialize Embeddings.



Returns:
serializes this Embeddings so that it can be reconstructed by `from_dict` class method.



Return type:
dict








BaseEmbeddings[¶](#baseembeddings "Link to this heading")
---------------------------------------------------------




*class* ibm\_watsonx\_ai.foundation\_models.embeddings.base\_embeddings.BaseEmbeddings[[source]](_modules/ibm_watsonx_ai/foundation_models/embeddings/base_embeddings.html#BaseEmbeddings)[¶](#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings "Link to this definition")
Bases: `ABC`


LangChain-like embedding function interface.




*abstract* embed\_documents(*texts*)[[source]](_modules/ibm_watsonx_ai/foundation_models/embeddings/base_embeddings.html#BaseEmbeddings.embed_documents)[¶](#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.embed_documents "Link to this definition")
Embed search docs.





*abstract* embed\_query(*text*)[[source]](_modules/ibm_watsonx_ai/foundation_models/embeddings/base_embeddings.html#BaseEmbeddings.embed_query)[¶](#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.embed_query "Link to this definition")
Embed query text.





*classmethod* from\_dict(*data*)[[source]](_modules/ibm_watsonx_ai/foundation_models/embeddings/base_embeddings.html#BaseEmbeddings.from_dict)[¶](#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.from_dict "Link to this definition")
Deserialize `BaseEmbeddings` into concrete one using arguments



Returns:
concrete Embeddings or None if data is incorrect



Return type:
[BaseEmbeddings](#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings "ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings") | None







to\_dict()[[source]](_modules/ibm_watsonx_ai/foundation_models/embeddings/base_embeddings.html#BaseEmbeddings.to_dict)[¶](#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.to_dict "Link to this definition")
Serialize Embeddings.



Returns:
serializes this Embeddings so that it can be reconstructed by `from_dict` class method.



Return type:
dict








Enums[¶](#enums "Link to this heading")
---------------------------------------




*class* metanames.EmbedTextParamsMetaNames[[source]](_modules/metanames.html#EmbedTextParamsMetaNames)[¶](#metanames.EmbedTextParamsMetaNames "Link to this definition")
Set of MetaNames for Foundation Model Embeddings Parameters.


Available MetaNames:





| MetaName | Type | Required | Example value |
| --- | --- | --- | --- |
| TRUNCATE\_INPUT\_TOKENS | int | N | `2` |
| RETURN\_OPTIONS | dict | N | `{'input_text': True}` |






*class* EmbeddingModels[¶](#EmbeddingModels "Link to this definition")
Bases: `Enum`


This represents a dynamically generated Enum for Embedding Models.


**Example of getting EmbeddingModels**



```
# GET EmbeddingModels ENUM
client.foundation_models.EmbeddingModels

# PRINT dict of Enums
client.foundation_models.EmbeddingModels.show()

```


**Example Output:**



```
{'SLATE_125M_ENGLISH_RTRVR': 'ibm/slate-125m-english-rtrvr',
...
'SLATE_30M_ENGLISH_RTRVR': 'ibm/slate-30m-english-rtrvr'}

```


**Example of initialising Embeddings with EmbeddingModels Enum:**



```
from ibm_watsonx_ai.foundation_models import Embeddings

embeddings = Embeddings(
    model_id=client.foundation_models.EmbeddingModels.SLATE_30M_ENGLISH_RTRVR,
    credentials=Credentials(...),
    project_id=project_id,
)

```





*class* ibm\_watsonx\_ai.foundation\_models.utils.enums.EmbeddingTypes(*value*)[[source]](_modules/ibm_watsonx_ai/foundation_models/utils/enums.html#EmbeddingTypes)[¶](#ibm_watsonx_ai.foundation_models.utils.enums.EmbeddingTypes "Link to this definition")
Bases: `Enum`



Deprecated since version 1.0.5: Use [`EmbeddingModels()`](#EmbeddingModels "EmbeddingModels") instead.



Supported embedding models.



Note


Current list of supported embeddings model types of various environments, user can check with
[`get_embeddings_model_specs()`](fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_embeddings_model_specs "ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_embeddings_model_specs")
or by referring to the [watsonx.ai](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models-embed.html?context=wx)
documentation.










[Next

Models](fm_models.html)
[Previous

Foundation Models](foundation_models.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Embeddings](#)
	+ [Embeddings](#id1)
		- [`Embeddings`](#ibm_watsonx_ai.foundation_models.embeddings.Embeddings)
			* [`Embeddings.embed_documents()`](#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.embed_documents)
			* [`Embeddings.embed_query()`](#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.embed_query)
			* [`Embeddings.generate()`](#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.generate)
			* [`Embeddings.to_dict()`](#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.to_dict)
	+ [BaseEmbeddings](#baseembeddings)
		- [`BaseEmbeddings`](#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings)
			* [`BaseEmbeddings.embed_documents()`](#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.embed_documents)
			* [`BaseEmbeddings.embed_query()`](#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.embed_query)
			* [`BaseEmbeddings.from_dict()`](#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.from_dict)
			* [`BaseEmbeddings.to_dict()`](#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.to_dict)
	+ [Enums](#enums)
		- [`EmbedTextParamsMetaNames`](#metanames.EmbedTextParamsMetaNames)
		- [`EmbeddingModels`](#EmbeddingModels)
		- [`EmbeddingTypes`](#ibm_watsonx_ai.foundation_models.utils.enums.EmbeddingTypes)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/fm_models.html








Models - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](#)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Models[¶](#models "Link to this heading")
=========================================


The `Model` module is an extension of `ModelInference` with langchain support (option to get WatsonxLLM wrapper for watsonx foundation models).



Modules[¶](#modules "Link to this heading")
-------------------------------------------



* [Model](fm_model.html)
	+ [`Model`](fm_model.html#ibm_watsonx_ai.foundation_models.Model)
		- [`Model.generate()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.generate)
		- [`Model.generate_text()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.generate_text)
		- [`Model.generate_text_stream()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.generate_text_stream)
		- [`Model.get_details()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.get_details)
		- [`Model.to_langchain()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.to_langchain)
		- [`Model.tokenize()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.tokenize)
	+ [Enums](fm_model.html#enums)
		- [`GenTextParamsMetaNames`](fm_model.html#metanames.GenTextParamsMetaNames)
		- [`GenTextReturnOptMetaNames`](fm_model.html#metanames.GenTextReturnOptMetaNames)
		- [`DecodingMethods`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods)
			* [`DecodingMethods.GREEDY`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods.GREEDY)
			* [`DecodingMethods.SAMPLE`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods.SAMPLE)
		- [`TextModels`](fm_model.html#TextModels)
		- [`ModelTypes`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes)
* [ModelInference](fm_model_inference.html)
	+ [`ModelInference`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference)
		- [`ModelInference.generate()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate)
		- [`ModelInference.generate_text()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate_text)
		- [`ModelInference.generate_text_stream()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate_text_stream)
		- [`ModelInference.get_details()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.get_details)
		- [`ModelInference.get_identifying_params()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.get_identifying_params)
		- [`ModelInference.to_langchain()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.to_langchain)
		- [`ModelInference.tokenize()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.tokenize)
* [`ModelInference` for Deployments](fm_deployments.html)
	+ [Infer text with deployments](fm_deployments.html#infer-text-with-deployments)
	+ [Creating `ModelInference` instance](fm_deployments.html#creating-modelinference-instance)
	+ [Generate methods](fm_deployments.html#generate-methods)









[Next

Model](fm_model.html)
[Previous

Embeddings](fm_embeddings.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Models](#)
	+ [Modules](#modules)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html








Model - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](#)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Model[¶](#model "Link to this heading")
=======================================




*class* ibm\_watsonx\_ai.foundation\_models.Model(*model\_id*, *credentials*, *params=None*, *project\_id=None*, *space\_id=None*, *verify=None*, *validate=True*)[[source]](_modules/ibm_watsonx_ai/foundation_models/model.html#Model)[¶](#ibm_watsonx_ai.foundation_models.Model "Link to this definition")
Bases: [`ModelInference`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference "ibm_watsonx_ai.foundation_models.inference.model_inference.ModelInference")


Instantiate the model interface.



Hint


To use the Model class with LangChain, use the [`to_langchain()`](#ibm_watsonx_ai.foundation_models.Model.to_langchain "ibm_watsonx_ai.foundation_models.Model.to_langchain") function.




Parameters:
* **model\_id** (*str*) – the type of model to use
* **credentials** (*ibm\_watsonx\_ai.Credentials* *or* *dict*) – credentials to Watson Machine Learning instance
* **params** (*dict**,* *optional*) – parameters to use during generate requests
* **project\_id** (*str**,* *optional*) – ID of the Watson Studio project
* **space\_id** (*str**,* *optional*) – ID of the Watson Studio space
* **verify** (*bool* *or* *str**,* *optional*) – user can pass as verify one of following:



	+ the path to a CA\_BUNDLE file
	+ the path of directory with certificates of trusted CAs
	+ True - default path to truststore will be taken
	+ False - no verification will be made
* **validate** (*bool**,* *optional*) – Model id validation, defaults to True





Note


One of these parameters is required: [‘project\_id ‘, ‘space\_id’]




Hint


You can copy the project\_id from the Project’s Manage tab (Project -> Manage -> General -> Details).



**Example**



```
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods

# To display example params enter
GenParams().get_example_values()

generate_params = {
    GenParams.MAX_NEW_TOKENS: 25
}

model = Model(
    model_id=ModelTypes.FLAN_UL2,
    params=generate_params,
    credentials=Credentials(
                    api_key = "***",
                    url = "https://us-south.ml.cloud.ibm.com"),
    project_id="*****"
    )

```




generate(*prompt=None*, *params=None*, *guardrails=False*, *guardrails\_hap\_params=None*, *guardrails\_pii\_params=None*, *concurrency\_limit=10*, *async\_mode=False*)[[source]](_modules/ibm_watsonx_ai/foundation_models/model.html#Model.generate)[¶](#ibm_watsonx_ai.foundation_models.Model.generate "Link to this definition")
Given a text prompt as input, and parameters the selected model (model\_id)
will generate a completion text as generated\_text.



Parameters:
* **params** (*dict*) – meta props for text generation, use `ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()` to view the list of MetaNames
* **concurrency\_limit** (*int*) – number of requests that will be sent in parallel, max is 10
* **prompt** (*str**,* *list*) – the prompt string or list of strings. If list of strings is passed requests will be managed in parallel with the rate of concurency\_limit
* **guardrails** (*bool*) – If True then potentially hateful, abusive, and/or profane language (HAP) detection
filter is toggle on for both prompt and generated text, defaults to False
* **guardrails\_hap\_params** – meta props for HAP moderations, use `ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()`
to view the list of MetaNames
* **async\_mode** (*bool*) – If True then yield results asynchronously (using generator). In this case both prompt and
generated text will be concatenated in the final response - under generated\_text, defaults
to False



Returns:
scoring result containing generated content



Return type:
dict




**Example**



```
q = "What is 1 + 1?"
generated_response = model.generate(prompt=q)
print(generated_response['results'][0]['generated_text'])

```





generate\_text(*prompt=None*, *params=None*, *raw\_response=False*, *guardrails=False*, *guardrails\_hap\_params=None*, *guardrails\_pii\_params=None*, *concurrency\_limit=10*)[[source]](_modules/ibm_watsonx_ai/foundation_models/model.html#Model.generate_text)[¶](#ibm_watsonx_ai.foundation_models.Model.generate_text "Link to this definition")
Given a text prompt as input, and parameters the selected model (model\_id)
will generate a completion text as generated\_text.



Parameters:
* **params** (*dict*) – meta props for text generation, use `ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()` to view the list of MetaNames
* **concurrency\_limit** (*int*) – number of requests that will be sent in parallel, max is 10
* **prompt** (*str**,* *list*) – the prompt string or list of strings. If list of strings is passed requests will be managed in parallel with the rate of concurency\_limit
* **raw\_response** (*bool**,* *optional*) – return the whole response object
* **guardrails** (*bool*) – If True then potentially hateful, abusive, and/or profane language (HAP) detection
filter is toggle on for both prompt and generated text, defaults to False
* **guardrails\_hap\_params** – meta props for HAP moderations, use `ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()`
to view the list of MetaNames



Returns:
generated content



Return type:
str or dict




**Example**



```
q = "What is 1 + 1?"
generated_text = model.generate_text(prompt=q)
print(generated_text)

```





generate\_text\_stream(*prompt=None*, *params=None*, *raw\_response=False*, *guardrails=False*, *guardrails\_hap\_params=None*, *guardrails\_pii\_params=None*)[[source]](_modules/ibm_watsonx_ai/foundation_models/model.html#Model.generate_text_stream)[¶](#ibm_watsonx_ai.foundation_models.Model.generate_text_stream "Link to this definition")
Given a text prompt as input, and parameters the selected model (model\_id)
will generate a streamed text as generate\_text\_stream.



Parameters:
* **params** (*dict*) – meta props for text generation, use `ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()` to view the list of MetaNames
* **prompt** (*str**,*) – the prompt string
* **raw\_response** (*bool**,* *optional*) – yields the whole response object
* **guardrails** (*bool*) – If True then potentially hateful, abusive, and/or profane language (HAP) detection
filter is toggle on for both prompt and generated text, defaults to False
* **guardrails\_hap\_params** – meta props for HAP moderations, use `ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()`
to view the list of MetaNames



Returns:
scoring result containing generated content



Return type:
generator




**Example**



```
q = "Write an epigram about the sun"
generated_response = model.generate_text_stream(prompt=q)

for chunk in generated_response:
    print(chunk, end='', flush=True)

```





get\_details()[[source]](_modules/ibm_watsonx_ai/foundation_models/model.html#Model.get_details)[¶](#ibm_watsonx_ai.foundation_models.Model.get_details "Link to this definition")
Get model’s details



Returns:
model’s details



Return type:
dict




**Example**



```
model.get_details()

```





to\_langchain()[[source]](_modules/ibm_watsonx_ai/foundation_models/model.html#Model.to_langchain)[¶](#ibm_watsonx_ai.foundation_models.Model.to_langchain "Link to this definition")

Returns:
WatsonxLLM wrapper for watsonx foundation models



Return type:
[WatsonxLLM](fm_extensions.html#langchain_ibm.WatsonxLLM "langchain_ibm.WatsonxLLM")




**Example**



```
from langchain import PromptTemplate
from langchain.chains import LLMChain
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

flan_ul2_model = Model(
    model_id=ModelTypes.FLAN_UL2,
    credentials=Credentials(
                api_key = "***",
                url = "https://us-south.ml.cloud.ibm.com"),
    project_id="*****"
    )

prompt_template = "What color is the {flower}?"

llm_chain = LLMChain(llm=flan_ul2_model.to_langchain(), prompt=PromptTemplate.from_template(prompt_template))
llm_chain('sunflower')

```





tokenize(*prompt*, *return\_tokens=False*)[[source]](_modules/ibm_watsonx_ai/foundation_models/model.html#Model.tokenize)[¶](#ibm_watsonx_ai.foundation_models.Model.tokenize "Link to this definition")
The text tokenize operation allows you to check the conversion of provided input to tokens for a given model.
It splits text into words or sub-words, which then are converted to ids through a look-up table (vocabulary).
Tokenization allows the model to have a reasonable vocabulary size.



Parameters:
* **prompt** (*str*) – the prompt string
* **return\_tokens** (*bool*) – the parameter for text tokenization, defaults to False



Returns:
the result of tokenizing the input string.



Return type:
dict




**Example**



```
q = "Write an epigram about the moon"
tokenized_response = model.tokenize(prompt=q, return_tokens=True)
print(tokenized_response["result"])

```





Enums[¶](#enums "Link to this heading")
---------------------------------------




*class* metanames.GenTextParamsMetaNames[[source]](_modules/metanames.html#GenTextParamsMetaNames)[¶](#metanames.GenTextParamsMetaNames "Link to this definition")
Set of MetaNames for Foundation Model Parameters.


Available MetaNames:





| MetaName | Type | Required | Example value |
| --- | --- | --- | --- |
| DECODING\_METHOD | str | N | `sample` |
| LENGTH\_PENALTY | dict | N | `{'decay_factor': 2.5, 'start_index': 5}` |
| TEMPERATURE | float | N | `0.5` |
| TOP\_P | float | N | `0.2` |
| TOP\_K | int | N | `1` |
| RANDOM\_SEED | int | N | `33` |
| REPETITION\_PENALTY | float | N | `2` |
| MIN\_NEW\_TOKENS | int | N | `50` |
| MAX\_NEW\_TOKENS | int | N | `200` |
| STOP\_SEQUENCES | list | N | `['fail']` |
| TIME\_LIMIT | int | N | `600000` |
| TRUNCATE\_INPUT\_TOKENS | int | N | `200` |
| PROMPT\_VARIABLES | dict | N | `{'object': 'brain'}` |
| RETURN\_OPTIONS | dict | N | `{'input_text': True, 'generated_tokens': True, 'input_tokens': True, 'token_logprobs': True, 'token_ranks': False, 'top_n_tokens': False}` |






*class* metanames.GenTextReturnOptMetaNames[[source]](_modules/metanames.html#GenTextReturnOptMetaNames)[¶](#metanames.GenTextReturnOptMetaNames "Link to this definition")
Set of MetaNames for Foundation Model Parameters.


Available MetaNames:





| MetaName | Type | Required | Example value |
| --- | --- | --- | --- |
| INPUT\_TEXT | bool | Y | `True` |
| GENERATED\_TOKENS | bool | N | `True` |
| INPUT\_TOKENS | bool | Y | `True` |
| TOKEN\_LOGPROBS | bool | N | `True` |
| TOKEN\_RANKS | bool | N | `True` |
| TOP\_N\_TOKENS | int | N | `True` |




Note


One of these parameters is required: [‘INPUT\_TEXT’, ‘INPUT\_TOKENS’]






*class* ibm\_watsonx\_ai.foundation\_models.utils.enums.DecodingMethods(*value*)[[source]](_modules/ibm_watsonx_ai/foundation_models/utils/enums.html#DecodingMethods)[¶](#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods "Link to this definition")
Bases: `Enum`


Supported decoding methods for text generation.




GREEDY *= 'greedy'*[¶](#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods.GREEDY "Link to this definition")



SAMPLE *= 'sample'*[¶](#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods.SAMPLE "Link to this definition")




*class* TextModels[¶](#TextModels "Link to this definition")
Bases: `Enum`


This represents a dynamically generated Enum for Foundation Models.


**Example of getting TextModels**



```
# GET TextModels ENUM
client.foundation_models.TextModels

# PRINT dict of Enums
client.foundation_models.TextModels.show()

```


**Example Output:**



```
{'CODELLAMA_34B_INSTRUCT_HF': 'codellama/codellama-34b-instruct-hf',
'FLAN_T5_XL': 'google/flan-t5-xl',
'FLAN_T5_XXL': 'google/flan-t5-xxl',
'FLAN_UL2': 'google/flan-ul2',
'GRANITE_13B_CHAT_V2': 'ibm/granite-13b-chat-v2',
'GRANITE_13B_INSTRUCT_V2': 'ibm/granite-13b-instruct-v2',
'GRANITE_20B_CODE_INSTRUCT': 'ibm/granite-20b-code-instruct',
'GRANITE_20B_MULTILINGUAL': 'ibm/granite-20b-multilingual',
'GRANITE_34B_CODE_INSTRUCT': 'ibm/granite-34b-code-instruct',
'GRANITE_3B_CODE_INSTRUCT': 'ibm/granite-3b-code-instruct',
'GRANITE_7B_LAB': 'ibm/granite-7b-lab',
...
'GRANITE_8B_CODE_INSTRUCT': 'ibm/granite-8b-code-instruct',
'LLAMA_2_13B_CHAT': 'meta-llama/llama-2-13b-chat',
'LLAMA_2_70B_CHAT': 'meta-llama/llama-2-70b-chat',
'LLAMA_3_70B_INSTRUCT': 'meta-llama/llama-3-70b-instruct',
'LLAMA_3_8B_INSTRUCT': 'meta-llama/llama-3-8b-instruct',
'MERLINITE_7B': 'ibm-mistralai/merlinite-7b',
'MIXTRAL_8X7B_INSTRUCT_V01': 'mistralai/mixtral-8x7b-instruct-v01',
'MIXTRAL_8X7B_INSTRUCT_V01_Q': 'ibm-mistralai/mixtral-8x7b-instruct-v01-q',
'MT0_XXL': 'bigscience/mt0-xxl'}

```


**Example of initialising ModelInference with TextModels Enum:**



```
from ibm_watsonx_ai.foundation_models import ModelInference

model = ModelInference(
    model_id=client.foundation_models.TextModels.GRANITE_13B_INSTRUCT_V2,
    credentials=Credentials(...),
    project_id=project_id,
)

```





*class* ibm\_watsonx\_ai.foundation\_models.utils.enums.ModelTypes(*value*)[[source]](_modules/ibm_watsonx_ai/foundation_models/utils/enums.html#ModelTypes)[¶](#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes "Link to this definition")
Bases: `Enum`



Deprecated since version 1.0.5: Use [`TextModels()`](#TextModels "TextModels") instead.



Supported foundation models.



Note


Current list of supported models types of various environments, user can check with
[`get_model_specs()`](fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_model_specs "ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_model_specs") or
by referring to the [watsonx.ai](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx)
documentation.










[Next

ModelInference](fm_model_inference.html)
[Previous

Models](fm_models.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Model](#)
	+ [`Model`](#ibm_watsonx_ai.foundation_models.Model)
		- [`Model.generate()`](#ibm_watsonx_ai.foundation_models.Model.generate)
		- [`Model.generate_text()`](#ibm_watsonx_ai.foundation_models.Model.generate_text)
		- [`Model.generate_text_stream()`](#ibm_watsonx_ai.foundation_models.Model.generate_text_stream)
		- [`Model.get_details()`](#ibm_watsonx_ai.foundation_models.Model.get_details)
		- [`Model.to_langchain()`](#ibm_watsonx_ai.foundation_models.Model.to_langchain)
		- [`Model.tokenize()`](#ibm_watsonx_ai.foundation_models.Model.tokenize)
	+ [Enums](#enums)
		- [`GenTextParamsMetaNames`](#metanames.GenTextParamsMetaNames)
		- [`GenTextReturnOptMetaNames`](#metanames.GenTextReturnOptMetaNames)
		- [`DecodingMethods`](#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods)
			* [`DecodingMethods.GREEDY`](#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods.GREEDY)
			* [`DecodingMethods.SAMPLE`](#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods.SAMPLE)
		- [`TextModels`](#TextModels)
		- [`ModelTypes`](#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/autoai_experiment.html








AutoAI experiment - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](#)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





AutoAI experiment[¶](#autoai-experiment "Link to this heading")
===============================================================



AutoAI[¶](#autoai "Link to this heading")
-----------------------------------------




*class* ibm\_watsonx\_ai.experiment.autoai.autoai.AutoAI(*credentials=None*, *project\_id=None*, *space\_id=None*, *verify=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/experiment/autoai/autoai.html#AutoAI)[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI "Link to this definition")
Bases: `BaseExperiment`


AutoAI class for pipeline models optimization automation.



Parameters:
* **credentials** (*dict*) – credentials to instance
* **project\_id** (*str**,* *optional*) – ID of the Watson Studio project
* **space\_id** (*str**,* *optional*) – ID of the Watson Studio Space
* **verify** (*bool* *or* *str**,* *optional*) – user can pass as verify one of following:



	+ the path to a CA\_BUNDLE file
	+ the path of directory with certificates of trusted CAs
	+ True - default path to truststore will be taken
	+ False - no verification will be made




**Example**



```
from ibm_watsonx_ai.experiment import AutoAI

experiment = AutoAI(
    credentials={
        "apikey": "...",
        "iam_apikey_description": "...",
        "iam_apikey_name": "...",
        "iam_role_crn": "...",
        "iam_serviceid_crn": "...",
        "instance_id": "...",
        "url": "https://us-south.ml.cloud.ibm.com"
    },
    project_id="...",
    space_id="...")

```




*class* ClassificationAlgorithms(*value*)[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms "Link to this definition")
Bases: `Enum`


Classification algorithms that AutoAI can use for IBM Cloud.




DT *= 'DecisionTreeClassifier'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.DT "Link to this definition")



EX\_TREES *= 'ExtraTreesClassifier'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.EX_TREES "Link to this definition")



GB *= 'GradientBoostingClassifier'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.GB "Link to this definition")



LGBM *= 'LGBMClassifier'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.LGBM "Link to this definition")



LR *= 'LogisticRegression'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.LR "Link to this definition")



RF *= 'RandomForestClassifier'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.RF "Link to this definition")



SnapBM *= 'SnapBoostingMachineClassifier'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapBM "Link to this definition")



SnapDT *= 'SnapDecisionTreeClassifier'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapDT "Link to this definition")



SnapLR *= 'SnapLogisticRegression'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapLR "Link to this definition")



SnapRF *= 'SnapRandomForestClassifier'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapRF "Link to this definition")



SnapSVM *= 'SnapSVMClassifier'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapSVM "Link to this definition")



XGB *= 'XGBClassifier'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.XGB "Link to this definition")




*class* DataConnectionTypes[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes "Link to this definition")
Bases: `object`


Supported types of DataConnection.




CA *= 'connection\_asset'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.CA "Link to this definition")



CN *= 'container'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.CN "Link to this definition")



DS *= 'data\_asset'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.DS "Link to this definition")



FS *= 'fs'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.FS "Link to this definition")



S3 *= 's3'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.S3 "Link to this definition")




*class* ForecastingAlgorithms(*value*)[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms "Link to this definition")
Bases: `Enum`


Forecasting algorithms that AutoAI can use for IBM watsonx.ai software with IBM Cloud Pak for Data.




ARIMA *= 'ARIMA'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.ARIMA "Link to this definition")



BATS *= 'BATS'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.BATS "Link to this definition")



ENSEMBLER *= 'Ensembler'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.ENSEMBLER "Link to this definition")



HW *= 'HoltWinters'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.HW "Link to this definition")



LR *= 'LinearRegression'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.LR "Link to this definition")



RF *= 'RandomForest'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.RF "Link to this definition")



SVM *= 'SVM'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.SVM "Link to this definition")




*class* Metrics[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics "Link to this definition")
Bases: `object`


Supported types of classification and regression metrics in AutoAI.




ACCURACY\_AND\_DISPARATE\_IMPACT\_SCORE *= 'accuracy\_and\_disparate\_impact'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ACCURACY_AND_DISPARATE_IMPACT_SCORE "Link to this definition")



ACCURACY\_SCORE *= 'accuracy'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ACCURACY_SCORE "Link to this definition")



AVERAGE\_PRECISION\_SCORE *= 'average\_precision'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.AVERAGE_PRECISION_SCORE "Link to this definition")



EXPLAINED\_VARIANCE\_SCORE *= 'explained\_variance'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.EXPLAINED_VARIANCE_SCORE "Link to this definition")



F1\_SCORE *= 'f1'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE "Link to this definition")



F1\_SCORE\_MACRO *= 'f1\_macro'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE_MACRO "Link to this definition")



F1\_SCORE\_MICRO *= 'f1\_micro'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE_MICRO "Link to this definition")



F1\_SCORE\_WEIGHTED *= 'f1\_weighted'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE_WEIGHTED "Link to this definition")



LOG\_LOSS *= 'neg\_log\_loss'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.LOG_LOSS "Link to this definition")



MEAN\_ABSOLUTE\_ERROR *= 'neg\_mean\_absolute\_error'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEAN_ABSOLUTE_ERROR "Link to this definition")



MEAN\_SQUARED\_ERROR *= 'neg\_mean\_squared\_error'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEAN_SQUARED_ERROR "Link to this definition")



MEAN\_SQUARED\_LOG\_ERROR *= 'neg\_mean\_squared\_log\_error'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEAN_SQUARED_LOG_ERROR "Link to this definition")



MEDIAN\_ABSOLUTE\_ERROR *= 'neg\_median\_absolute\_error'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEDIAN_ABSOLUTE_ERROR "Link to this definition")



PRECISION\_SCORE *= 'precision'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE "Link to this definition")



PRECISION\_SCORE\_MACRO *= 'precision\_macro'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE_MACRO "Link to this definition")



PRECISION\_SCORE\_MICRO *= 'precision\_micro'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE_MICRO "Link to this definition")



PRECISION\_SCORE\_WEIGHTED *= 'precision\_weighted'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE_WEIGHTED "Link to this definition")



R2\_AND\_DISPARATE\_IMPACT\_SCORE *= 'r2\_and\_disparate\_impact'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.R2_AND_DISPARATE_IMPACT_SCORE "Link to this definition")



R2\_SCORE *= 'r2'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.R2_SCORE "Link to this definition")



RECALL\_SCORE *= 'recall'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE "Link to this definition")



RECALL\_SCORE\_MACRO *= 'recall\_macro'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE_MACRO "Link to this definition")



RECALL\_SCORE\_MICRO *= 'recall\_micro'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE_MICRO "Link to this definition")



RECALL\_SCORE\_WEIGHTED *= 'recall\_weighted'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE_WEIGHTED "Link to this definition")



ROC\_AUC\_SCORE *= 'roc\_auc'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ROC_AUC_SCORE "Link to this definition")



ROOT\_MEAN\_SQUARED\_ERROR *= 'neg\_root\_mean\_squared\_error'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ROOT_MEAN_SQUARED_ERROR "Link to this definition")



ROOT\_MEAN\_SQUARED\_LOG\_ERROR *= 'neg\_root\_mean\_squared\_log\_error'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ROOT_MEAN_SQUARED_LOG_ERROR "Link to this definition")




*class* PipelineTypes[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PipelineTypes "Link to this definition")
Bases: `object`


Supported types of Pipelines.




LALE *= 'lale'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PipelineTypes.LALE "Link to this definition")



SKLEARN *= 'sklearn'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PipelineTypes.SKLEARN "Link to this definition")




*class* PredictionType[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType "Link to this definition")
Bases: `object`


Supported types of learning.




BINARY *= 'binary'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.BINARY "Link to this definition")



CLASSIFICATION *= 'classification'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.CLASSIFICATION "Link to this definition")



FORECASTING *= 'forecasting'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.FORECASTING "Link to this definition")



MULTICLASS *= 'multiclass'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.MULTICLASS "Link to this definition")



REGRESSION *= 'regression'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.REGRESSION "Link to this definition")



TIMESERIES\_ANOMALY\_PREDICTION *= 'timeseries\_anomaly\_prediction'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.TIMESERIES_ANOMALY_PREDICTION "Link to this definition")




*class* RegressionAlgorithms(*value*)[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms "Link to this definition")
Bases: `Enum`


Regression algorithms that AutoAI can use for IBM Cloud.




DT *= 'DecisionTreeRegressor'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.DT "Link to this definition")



EX\_TREES *= 'ExtraTreesRegressor'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.EX_TREES "Link to this definition")



GB *= 'GradientBoostingRegressor'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.GB "Link to this definition")



LGBM *= 'LGBMRegressor'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.LGBM "Link to this definition")



LR *= 'LinearRegression'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.LR "Link to this definition")



RF *= 'RandomForestRegressor'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.RF "Link to this definition")



RIDGE *= 'Ridge'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.RIDGE "Link to this definition")



SnapBM *= 'SnapBoostingMachineRegressor'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.SnapBM "Link to this definition")



SnapDT *= 'SnapDecisionTreeRegressor'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.SnapDT "Link to this definition")



SnapRF *= 'SnapRandomForestRegressor'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.SnapRF "Link to this definition")



XGB *= 'XGBRegressor'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.XGB "Link to this definition")




*class* SamplingTypes[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes "Link to this definition")
Bases: `object`


Types of training data sampling.




FIRST\_VALUES *= 'first\_n\_records'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.FIRST_VALUES "Link to this definition")



LAST\_VALUES *= 'truncate'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.LAST_VALUES "Link to this definition")



RANDOM *= 'random'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.RANDOM "Link to this definition")



STRATIFIED *= 'stratified'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.STRATIFIED "Link to this definition")




*class* TShirtSize[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize "Link to this definition")
Bases: `object`


Possible sizes of the AutoAI POD.
Depending on the POD size, AutoAI can support different data set sizes.


* S - small (2vCPUs and 8GB of RAM)
* M - Medium (4vCPUs and 16GB of RAM)
* L - Large (8vCPUs and 32GB of RAM))
* XL - Extra Large (16vCPUs and 64GB of RAM)




L *= 'l'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.L "Link to this definition")



M *= 'm'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.M "Link to this definition")



S *= 's'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.S "Link to this definition")



XL *= 'xl'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.XL "Link to this definition")




*class* Transformers[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers "Link to this definition")
Bases: `object`


Supported types of congito transformers names in AutoAI.




ABS *= 'abs'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.ABS "Link to this definition")



CBRT *= 'cbrt'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.CBRT "Link to this definition")



COS *= 'cos'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.COS "Link to this definition")



CUBE *= 'cube'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.CUBE "Link to this definition")



DIFF *= 'diff'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.DIFF "Link to this definition")



DIVIDE *= 'divide'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.DIVIDE "Link to this definition")



FEATUREAGGLOMERATION *= 'featureagglomeration'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.FEATUREAGGLOMERATION "Link to this definition")



ISOFORESTANOMALY *= 'isoforestanomaly'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.ISOFORESTANOMALY "Link to this definition")



LOG *= 'log'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.LOG "Link to this definition")



MAX *= 'max'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.MAX "Link to this definition")



MINMAXSCALER *= 'minmaxscaler'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.MINMAXSCALER "Link to this definition")



NXOR *= 'nxor'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.NXOR "Link to this definition")



PCA *= 'pca'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.PCA "Link to this definition")



PRODUCT *= 'product'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.PRODUCT "Link to this definition")



ROUND *= 'round'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.ROUND "Link to this definition")



SIGMOID *= 'sigmoid'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SIGMOID "Link to this definition")



SIN *= 'sin'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SIN "Link to this definition")



SQRT *= 'sqrt'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SQRT "Link to this definition")



SQUARE *= 'square'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SQUARE "Link to this definition")



STDSCALER *= 'stdscaler'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.STDSCALER "Link to this definition")



SUM *= 'sum'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SUM "Link to this definition")



TAN *= 'tan'*[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.TAN "Link to this definition")




optimizer(*name*, *\**, *prediction\_type*, *prediction\_column=None*, *prediction\_columns=None*, *timestamp\_column\_name=None*, *scoring=None*, *desc=None*, *test\_size=None*, *holdout\_size=None*, *max\_number\_of\_estimators=None*, *train\_sample\_rows\_test\_size=None*, *include\_only\_estimators=None*, *daub\_include\_only\_estimators=None*, *include\_batched\_ensemble\_estimators=None*, *backtest\_num=None*, *lookback\_window=None*, *forecast\_window=None*, *backtest\_gap\_length=None*, *feature\_columns=None*, *pipeline\_types=None*, *supporting\_features\_at\_forecast=None*, *cognito\_transform\_names=None*, *csv\_separator=','*, *excel\_sheet=None*, *encoding='utf-8'*, *positive\_label=None*, *drop\_duplicates=True*, *outliers\_columns=None*, *text\_processing=None*, *word2vec\_feature\_number=None*, *daub\_give\_priority\_to\_runtime=None*, *fairness\_info=None*, *sampling\_type=None*, *sample\_size\_limit=None*, *sample\_rows\_limit=None*, *sample\_percentage\_limit=None*, *n\_parallel\_data\_connections=None*, *number\_of\_batch\_rows=None*, *categorical\_imputation\_strategy=None*, *numerical\_imputation\_strategy=None*, *numerical\_imputation\_value=None*, *imputation\_threshold=None*, *retrain\_on\_holdout=None*, *categorical\_columns=None*, *numerical\_columns=None*, *test\_data\_csv\_separator=','*, *test\_data\_excel\_sheet=None*, *test\_data\_encoding='utf-8'*, *confidence\_level=None*, *incremental\_learning=None*, *early\_stop\_enabled=None*, *early\_stop\_window\_size=None*, *time\_ordered\_data=None*, *feature\_selector\_mode=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/experiment/autoai/autoai.html#AutoAI.optimizer)[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.optimizer "Link to this definition")
Initialize an AutoAi optimizer.



Parameters:
* **name** (*str*) – name for the AutoPipelines
* **prediction\_type** ([*PredictionType*](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType "ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType")) – type of the prediction
* **prediction\_column** (*str**,* *optional*) – name of the target/label column, required for multiclass, binary and regression
prediction types
* **prediction\_columns** (*list**[**str**]**,* *optional*) – names of the target/label columns, required for forecasting prediction type
* **timestamp\_column\_name** (*str**,* *optional*) – name of timestamp column for time series forecasting
* **scoring** ([*Metrics*](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics "ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics")*,* *optional*) – type of the metric to optimize with, not used for forecasting
* **desc** (*str**,* *optional*) – description
* **test\_size** – deprecated, use holdout\_size instead
* **holdout\_size** (*float**,* *optional*) – percentage of the entire dataset to leave as a holdout
* **max\_number\_of\_estimators** (*int**,* *optional*) – maximum number (top-K ranked by DAUB model selection)
of the selected algorithm, or estimator types, for example LGBMClassifierEstimator,
XGBoostClassifierEstimator, or LogisticRegressionEstimator to use in pipeline composition,
the default is None that means the true default value will be determined by
the internal different algorithms, where only the highest ranked by model selection algorithm type is used
* **train\_sample\_rows\_test\_size** (*float**,* *optional*) – training data sampling percentage
* **daub\_include\_only\_estimators** – deprecated, use include\_only\_estimators instead
* **include\_batched\_ensemble\_estimators** (*list**[**BatchedClassificationAlgorithms* *or* *BatchedRegressionAlgorithms**]**,* *optional*) – list of batched ensemble estimators to include
in computation process, see: AutoAI.BatchedClassificationAlgorithms, AutoAI.BatchedRegressionAlgorithms
* **include\_only\_estimators** (*List**[*[*ClassificationAlgorithms*](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms "ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms") *or* [*RegressionAlgorithms*](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms "ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms") *or* [*ForecastingAlgorithms*](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms "ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms")*]**]**,* *optional*) – list of estimators to include in computation process, see:
AutoAI.ClassificationAlgorithms, AutoAI.RegressionAlgorithms or AutoAI.ForecastingAlgorithms
* **backtest\_num** (*int**,* *optional*) – number of backtests used for forecasting prediction type, default value: 4,
value from range [0, 20]
* **lookback\_window** (*int**,* *optional*) – length of lookback window used for forecasting prediction type,
default value: 10, if set to -1 lookback window will be auto-detected
* **forecast\_window** (*int**,* *optional*) – length of forecast window used for forecasting prediction type, default value: 1,
value from range [1, 60]
* **backtest\_gap\_length** (*int**,* *optional*) – gap between backtests used for forecasting prediction type,
default value: 0, value from range [0, data length / 4]
* **feature\_columns** (*list**[**str**]**,* *optional*) – list of feature columns used for forecasting prediction type,
may contain target column and/or supporting feature columns, list of columns to be detected whether there are anomalies for timeseries anomaly prediction type
* **pipeline\_types** (*list**[*[*ForecastingPipelineTypes*](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes "ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes") *or* [*TimeseriesAnomalyPredictionPipelineTypes*](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes "ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes")*]**,* *optional*) – list of pipeline types to be used for forecasting or timeseries anomaly prediction type
* **supporting\_features\_at\_forecast** (*bool**,* *optional*) – enables usage of future supporting feature values during forecast
* **cognito\_transform\_names** (*list**[*[*Transformers*](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers "ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers")*]**,* *optional*) – list of transformers to include in the feature enginnering computation process,
see: AutoAI.Transformers
* **csv\_separator** (*list**[**str**] or* *str**,* *optional*) – the separator, or list of separators to try for separating columns in a CSV file,
not used if the file\_name is not a CSV file, default is ‘,’
* **excel\_sheet** (*str**,* *optional*) – name of the excel sheet to use, only applicable when xlsx file is an input,
support for number of the sheet is deprecated, by default first sheet is used
* **encoding** (*str**,* *optional*) – encoding type for CSV training file
* **positive\_label** (*str**,* *optional*) – the positive class to report when binary classification, when multiclass or regression,
this will be ignored
* **t\_shirt\_size** ([*TShirtSize*](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize "ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize")*,* *optional*) – the size of the remote AutoAI POD instance (computing resources),
only applicable to a remote scenario, see: AutoAI.TShirtSize
* **drop\_duplicates** (*bool**,* *optional*) – if True duplicated rows in data will be removed before further processing
* **outliers\_columns** (*list**,* *optional*) – replace outliers with NaN using IQR method for specified columns. By default,
turned ON for regression learning\_type and target column. To turn OFF empty list of columns must be passed
* **text\_processing** (*bool**,* *optional*) – if True text processing will be enabled, applicable only on Cloud
* **word2vec\_feature\_number** (*int**,* *optional*) – number of features which will be generated from text column,
will be applied only if text\_processing is True, if None the default value will be taken
* **daub\_give\_priority\_to\_runtime** (*float**,* *optional*) – the importance of run time over score for pipelines ranking,
can take values between 0 and 5, if set to 0.0 only score is used,
if set to 1 equally score and runtime are used, if set to value higher than 1
the runtime gets higher importance over score
* **fairness\_info** (*fairness\_info*) – dictionary that specifies metadata needed for measuring fairness,
it contains three key values: favorable\_labels, unfavorable\_labels and protected\_attributes,
the favorable\_labels attribute indicates that when the class column contains one of the value from list,
that is considered a positive outcome, the unfavorable\_labels is oposite to the favorable\_labels
and is obligatory for regression learning type, a protected attribute is a list of features that partition
the population into groups whose outcome should have parity, if protected attribute is empty list
then automatic detection of protected attributes will be run,
if fairness\_info is passed then fairness metric will be calculated
* **n\_parallel\_data\_connections** (*int**,* *optional*) – number of maximum parallel connection to data source,
supported only for IBM Cloud Pak® for Data 4.0.1 and above
* **categorical\_imputation\_strategy** ([*ImputationStrategy*](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy "ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy")*,* *optional*) – missing values imputation strategy for categorical columns


Possible values (only non-forecasting scenario):



	+ ImputationStrategy.MEAN
	+ ImputationStrategy.MEDIAN
	+ ImputationStrategy.MOST\_FREQUENT (default)
* **numerical\_imputation\_strategy** – missing values imputation strategy for numerical columns


Possible values (non-forecasting scenario):



	+ ImputationStrategy.MEAN
	+ ImputationStrategy.MEDIAN (default)
	+ ImputationStrategy.MOST\_FREQUENT
Possible values (forecasting scenario):



	+ ImputationStrategy.MEAN
	+ ImputationStrategy.MEDIAN
	+ ImputationStrategy.BEST\_OF\_DEFAULT\_IMPUTERS (default)
	+ ImputationStrategy.VALUE
	+ ImputationStrategy.FLATTEN\_ITERATIVE
	+ ImputationStrategy.LINEAR
	+ ImputationStrategy.CUBIC
	+ ImputationStrategy.PREVIOUS
	+ ImputationStrategy.NEXT
	+ ImputationStrategy.NO\_IMPUTATION
* **numerical\_imputation\_value** (*float**,* *optional*) – value for filling missing values if numerical\_imputation\_strategy
is set to ImputationStrategy.VALUE, for forecasting only
* **imputation\_threshold** (*float**,* *optional*) – maximum threshold of missing values imputation, for forecasting only
* **retrain\_on\_holdout** (*bool**,* *optional*) – if True final pipelines will be train also on holdout data
* **categorical\_columns** (*list**,* *optional*) – list of columns names that must be treated as categorical
* **numerical\_columns** (*list**,* *optional*) – list of columns names that must be treated as numerical
* **sampling\_type** (*str**,* *optional*) – type of sampling data for training, one of SamplingTypes enum values,
default is SamplingTypes.FIRST\_N\_RECORDS, supported only for IBM Cloud Pak® for Data 4.0.1 and above
* **sample\_size\_limit** (*int**,* *optional*) – the size of sample upper bound (in bytes). The default value is 1GB,
supported only for IBM Cloud Pak® for Data 4.5 and above
* **sample\_rows\_limit** (*int**,* *optional*) – the size of sample upper bound (in rows),
supported only for IBM Cloud Pak® for Data 4.6 and above
* **sample\_percentage\_limit** (*float**,* *optional*) – the size of sample upper bound (as fraction of dataset size),
supported only for IBM Cloud Pak® for Data 4.6 and above
* **number\_of\_batch\_rows** (*int**,* *optional*) – number of rows to read in each batch when reading from flight connection
* **test\_data\_csv\_separator** (*list**[**str**] or* *str**,* *optional*) – the separator, or list of separators to try for separating
columns in a CSV user-defined holdout/test file, not used if the file\_name is not a CSV file,
default is ‘,’
* **test\_data\_excel\_sheet** (*str* *or* *int**,* *optional*) – name of the excel sheet to use for user-defined holdout/test data,
only use when xlsx file is an test, dataset file, by default first sheet is used
* **test\_data\_encoding** (*str**,* *optional*) – encoding type for CSV user-defined holdout/test file
* **confidence\_level** (*float**,* *optional*) – when the pipeline “PointwiseBoundedHoltWinters” or “PointwiseBoundedBATS” is used,
the prediction interval is calculated at a given confidence\_level to decide if a data record
is an anomaly or not, optional for timeseries anomaly prediction
* **incremental\_learning** (*bool**,* *optional*) – triggers incremental learning process for supported pipelines
* **early\_stop\_enabled** (*bool**,* *optional*) – enables early stop for incremental learning process
* **early\_stop\_window\_size** (*int**,* *optional*) – the number of iterations without score improvements before training stop
* **time\_ordered\_data** (*bool**,* *optional*) – defines user preference about time-based analise. If True, the analysis will
consider the data as time-ordered and time-based. Supported only for regression.
* **feature\_selector\_mode** (*str**,* *optional*) – defines if feature selector should be triggered [“on”, “off”, “auto”].
The “auto” mode analyses the impact of removal of insignificant features. If there is drop in accuracy,
the PCA is applied to insignificant features. Principal components describing variance in 30% or higher
are selected in place of insignificant features. The model is evaluated again. If there is still drop
in accuracy all features are used.
The “on” mode removes all insignificant features (0.0. importance). Feature selector is applied during
cognito phase (applicable to pipelines with feature engineering stage).



Returns:
RemoteAutoPipelines or LocalAutoPipelines, depends on how you initialize the AutoAI object



Return type:
RemoteAutoPipelines or LocalAutoPipelines




**Examples**



```
from ibm_watsonx_ai.experiment import AutoAI
experiment = AutoAI(...)

fairness_info = {
           "protected_attributes": [
               {"feature": "Sex", "reference_group": ['male'], "monitored_group": ['female']},
               {"feature": "Age", "reference_group": [[50,60]], "monitored_group": [[18, 49]]}
           ],
           "favorable_labels": ["No Risk"],
           "unfavorable_labels": ["Risk"],
           }

optimizer = experiment.optimizer(
       name="name of the optimizer.",
       prediction_type=AutoAI.PredictionType.BINARY,
       prediction_column="y",
       scoring=AutoAI.Metrics.ROC_AUC_SCORE,
       desc="Some description.",
       holdout_size=0.1,
       max_number_of_estimators=1,
       fairness_info= fairness_info,
       cognito_transform_names=[AutoAI.Transformers.SUM,AutoAI.Transformers.MAX],
       train_sample_rows_test_size=1,
       include_only_estimators=[AutoAI.ClassificationAlgorithms.LGBM, AutoAI.ClassificationAlgorithms.XGB],
       t_shirt_size=AutoAI.TShirtSize.L
   )

optimizer = experiment.optimizer(
       name="name of the optimizer.",
       prediction_type=AutoAI.PredictionType.MULTICLASS,
       prediction_column="y",
       scoring=AutoAI.Metrics.ROC_AUC_SCORE,
       desc="Some description.",
   )

```





runs(*\**, *filter*)[[source]](_modules/ibm_watsonx_ai/experiment/autoai/autoai.html#AutoAI.runs)[¶](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.runs "Link to this definition")
Get the historical runs but with Pipeline name filter (for remote scenario).
Get the historical runs but with experiment name filter (for local scenario).



Parameters:
**filter** (*str*) – Pipeline name to filter the historical runs or experiment name to filter
the local historical runs



Returns:
object managing the list of runs



Return type:
AutoPipelinesRuns or LocalAutoPipelinesRuns




**Example**



```
from ibm_watsonx_ai.experiment import AutoAI

experiment = AutoAI(...)
experiment.runs(filter='Test').list()

```










[Next

Deployment Modules for AutoAI models](autoai_deployment_modules.html)
[Previous

Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [AutoAI experiment](#)
	+ [AutoAI](#autoai)
		- [`AutoAI`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI)
			* [`AutoAI.ClassificationAlgorithms`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms)
				+ [`AutoAI.ClassificationAlgorithms.DT`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.DT)
				+ [`AutoAI.ClassificationAlgorithms.EX_TREES`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.EX_TREES)
				+ [`AutoAI.ClassificationAlgorithms.GB`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.GB)
				+ [`AutoAI.ClassificationAlgorithms.LGBM`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.LGBM)
				+ [`AutoAI.ClassificationAlgorithms.LR`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.LR)
				+ [`AutoAI.ClassificationAlgorithms.RF`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.RF)
				+ [`AutoAI.ClassificationAlgorithms.SnapBM`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapBM)
				+ [`AutoAI.ClassificationAlgorithms.SnapDT`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapDT)
				+ [`AutoAI.ClassificationAlgorithms.SnapLR`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapLR)
				+ [`AutoAI.ClassificationAlgorithms.SnapRF`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapRF)
				+ [`AutoAI.ClassificationAlgorithms.SnapSVM`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapSVM)
				+ [`AutoAI.ClassificationAlgorithms.XGB`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.XGB)
			* [`AutoAI.DataConnectionTypes`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes)
				+ [`AutoAI.DataConnectionTypes.CA`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.CA)
				+ [`AutoAI.DataConnectionTypes.CN`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.CN)
				+ [`AutoAI.DataConnectionTypes.DS`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.DS)
				+ [`AutoAI.DataConnectionTypes.FS`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.FS)
				+ [`AutoAI.DataConnectionTypes.S3`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.S3)
			* [`AutoAI.ForecastingAlgorithms`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms)
				+ [`AutoAI.ForecastingAlgorithms.ARIMA`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.ARIMA)
				+ [`AutoAI.ForecastingAlgorithms.BATS`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.BATS)
				+ [`AutoAI.ForecastingAlgorithms.ENSEMBLER`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.ENSEMBLER)
				+ [`AutoAI.ForecastingAlgorithms.HW`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.HW)
				+ [`AutoAI.ForecastingAlgorithms.LR`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.LR)
				+ [`AutoAI.ForecastingAlgorithms.RF`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.RF)
				+ [`AutoAI.ForecastingAlgorithms.SVM`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.SVM)
			* [`AutoAI.Metrics`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics)
				+ [`AutoAI.Metrics.ACCURACY_AND_DISPARATE_IMPACT_SCORE`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ACCURACY_AND_DISPARATE_IMPACT_SCORE)
				+ [`AutoAI.Metrics.ACCURACY_SCORE`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ACCURACY_SCORE)
				+ [`AutoAI.Metrics.AVERAGE_PRECISION_SCORE`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.AVERAGE_PRECISION_SCORE)
				+ [`AutoAI.Metrics.EXPLAINED_VARIANCE_SCORE`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.EXPLAINED_VARIANCE_SCORE)
				+ [`AutoAI.Metrics.F1_SCORE`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE)
				+ [`AutoAI.Metrics.F1_SCORE_MACRO`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE_MACRO)
				+ [`AutoAI.Metrics.F1_SCORE_MICRO`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE_MICRO)
				+ [`AutoAI.Metrics.F1_SCORE_WEIGHTED`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE_WEIGHTED)
				+ [`AutoAI.Metrics.LOG_LOSS`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.LOG_LOSS)
				+ [`AutoAI.Metrics.MEAN_ABSOLUTE_ERROR`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEAN_ABSOLUTE_ERROR)
				+ [`AutoAI.Metrics.MEAN_SQUARED_ERROR`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEAN_SQUARED_ERROR)
				+ [`AutoAI.Metrics.MEAN_SQUARED_LOG_ERROR`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEAN_SQUARED_LOG_ERROR)
				+ [`AutoAI.Metrics.MEDIAN_ABSOLUTE_ERROR`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEDIAN_ABSOLUTE_ERROR)
				+ [`AutoAI.Metrics.PRECISION_SCORE`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE)
				+ [`AutoAI.Metrics.PRECISION_SCORE_MACRO`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE_MACRO)
				+ [`AutoAI.Metrics.PRECISION_SCORE_MICRO`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE_MICRO)
				+ [`AutoAI.Metrics.PRECISION_SCORE_WEIGHTED`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE_WEIGHTED)
				+ [`AutoAI.Metrics.R2_AND_DISPARATE_IMPACT_SCORE`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.R2_AND_DISPARATE_IMPACT_SCORE)
				+ [`AutoAI.Metrics.R2_SCORE`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.R2_SCORE)
				+ [`AutoAI.Metrics.RECALL_SCORE`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE)
				+ [`AutoAI.Metrics.RECALL_SCORE_MACRO`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE_MACRO)
				+ [`AutoAI.Metrics.RECALL_SCORE_MICRO`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE_MICRO)
				+ [`AutoAI.Metrics.RECALL_SCORE_WEIGHTED`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE_WEIGHTED)
				+ [`AutoAI.Metrics.ROC_AUC_SCORE`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ROC_AUC_SCORE)
				+ [`AutoAI.Metrics.ROOT_MEAN_SQUARED_ERROR`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ROOT_MEAN_SQUARED_ERROR)
				+ [`AutoAI.Metrics.ROOT_MEAN_SQUARED_LOG_ERROR`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ROOT_MEAN_SQUARED_LOG_ERROR)
			* [`AutoAI.PipelineTypes`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PipelineTypes)
				+ [`AutoAI.PipelineTypes.LALE`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PipelineTypes.LALE)
				+ [`AutoAI.PipelineTypes.SKLEARN`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PipelineTypes.SKLEARN)
			* [`AutoAI.PredictionType`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType)
				+ [`AutoAI.PredictionType.BINARY`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.BINARY)
				+ [`AutoAI.PredictionType.CLASSIFICATION`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.CLASSIFICATION)
				+ [`AutoAI.PredictionType.FORECASTING`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.FORECASTING)
				+ [`AutoAI.PredictionType.MULTICLASS`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.MULTICLASS)
				+ [`AutoAI.PredictionType.REGRESSION`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.REGRESSION)
				+ [`AutoAI.PredictionType.TIMESERIES_ANOMALY_PREDICTION`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.TIMESERIES_ANOMALY_PREDICTION)
			* [`AutoAI.RegressionAlgorithms`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms)
				+ [`AutoAI.RegressionAlgorithms.DT`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.DT)
				+ [`AutoAI.RegressionAlgorithms.EX_TREES`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.EX_TREES)
				+ [`AutoAI.RegressionAlgorithms.GB`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.GB)
				+ [`AutoAI.RegressionAlgorithms.LGBM`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.LGBM)
				+ [`AutoAI.RegressionAlgorithms.LR`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.LR)
				+ [`AutoAI.RegressionAlgorithms.RF`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.RF)
				+ [`AutoAI.RegressionAlgorithms.RIDGE`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.RIDGE)
				+ [`AutoAI.RegressionAlgorithms.SnapBM`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.SnapBM)
				+ [`AutoAI.RegressionAlgorithms.SnapDT`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.SnapDT)
				+ [`AutoAI.RegressionAlgorithms.SnapRF`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.SnapRF)
				+ [`AutoAI.RegressionAlgorithms.XGB`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.XGB)
			* [`AutoAI.SamplingTypes`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes)
				+ [`AutoAI.SamplingTypes.FIRST_VALUES`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.FIRST_VALUES)
				+ [`AutoAI.SamplingTypes.LAST_VALUES`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.LAST_VALUES)
				+ [`AutoAI.SamplingTypes.RANDOM`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.RANDOM)
				+ [`AutoAI.SamplingTypes.STRATIFIED`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.STRATIFIED)
			* [`AutoAI.TShirtSize`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize)
				+ [`AutoAI.TShirtSize.L`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.L)
				+ [`AutoAI.TShirtSize.M`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.M)
				+ [`AutoAI.TShirtSize.S`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.S)
				+ [`AutoAI.TShirtSize.XL`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.XL)
			* [`AutoAI.Transformers`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers)
				+ [`AutoAI.Transformers.ABS`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.ABS)
				+ [`AutoAI.Transformers.CBRT`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.CBRT)
				+ [`AutoAI.Transformers.COS`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.COS)
				+ [`AutoAI.Transformers.CUBE`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.CUBE)
				+ [`AutoAI.Transformers.DIFF`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.DIFF)
				+ [`AutoAI.Transformers.DIVIDE`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.DIVIDE)
				+ [`AutoAI.Transformers.FEATUREAGGLOMERATION`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.FEATUREAGGLOMERATION)
				+ [`AutoAI.Transformers.ISOFORESTANOMALY`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.ISOFORESTANOMALY)
				+ [`AutoAI.Transformers.LOG`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.LOG)
				+ [`AutoAI.Transformers.MAX`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.MAX)
				+ [`AutoAI.Transformers.MINMAXSCALER`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.MINMAXSCALER)
				+ [`AutoAI.Transformers.NXOR`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.NXOR)
				+ [`AutoAI.Transformers.PCA`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.PCA)
				+ [`AutoAI.Transformers.PRODUCT`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.PRODUCT)
				+ [`AutoAI.Transformers.ROUND`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.ROUND)
				+ [`AutoAI.Transformers.SIGMOID`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SIGMOID)
				+ [`AutoAI.Transformers.SIN`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SIN)
				+ [`AutoAI.Transformers.SQRT`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SQRT)
				+ [`AutoAI.Transformers.SQUARE`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SQUARE)
				+ [`AutoAI.Transformers.STDSCALER`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.STDSCALER)
				+ [`AutoAI.Transformers.SUM`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SUM)
				+ [`AutoAI.Transformers.TAN`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.TAN)
			* [`AutoAI.optimizer()`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.optimizer)
			* [`AutoAI.runs()`](#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.runs)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/setup.html








Setup - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](#)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Setup[¶](#setup "Link to this heading")
=======================================


The setup of watsonx.ai client might differ depending on the product offering. Choose an offering option from the list below
to see setup details.



* [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [Requirements](setup_cloud.html#requirements)
	+ [Supported machine learning frameworks](setup_cloud.html#supported-machine-learning-frameworks)
	+ [Authentication](setup_cloud.html#authentication)
	+ [Firewall settings](setup_cloud.html#firewall-settings)
* [IBM watsonx.ai software](setup_cpd.html)
	+ [Requirements](setup_cpd.html#requirements)
	+ [Supported machine learning frameworks](setup_cpd.html#supported-machine-learning-frameworks)
	+ [Authentication](setup_cpd.html#authentication)








[Next

IBM watsonx.ai for IBM Cloud](setup_cloud.html)
[Previous

Installation](install.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)
















# Text from https://ibm.github.io/watsonx-ai-python-sdk/pt_working_with_class_and_prompt_tuner.html








Working with TuneExperiment and PromptTuner - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](#)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Working with TuneExperiment and PromptTuner[¶](#working-with-tuneexperiment-and-prompttuner "Link to this heading")
===================================================================================================================


This section shows how to creating tuning experiment, deploy model and use ModelInference module with created deployment.



* [Tune Experiment run](pt_tune_experiment_run.html)
	+ [Configure PromptTuner](pt_tune_experiment_run.html#configure-prompttuner)
	+ [Get configuration parameters](pt_tune_experiment_run.html#get-configuration-parameters)
	+ [Run prompt tuning](pt_tune_experiment_run.html#run-prompt-tuning)
	+ [Get run status, get run details](pt_tune_experiment_run.html#get-run-status-get-run-details)
	+ [Get data connections](pt_tune_experiment_run.html#get-data-connections)
	+ [Summary](pt_tune_experiment_run.html#summary)
	+ [Plot learning curves](pt_tune_experiment_run.html#plot-learning-curves)
	+ [Get model identifier](pt_tune_experiment_run.html#get-model-identifier)
* [Tuned Model Inference](pt_model_inference.html)
	+ [Working with deployments](pt_model_inference.html#working-with-deployments)
	+ [Creating `ModelInference` instance](pt_model_inference.html#creating-modelinference-instance)
	+ [Importing data](pt_model_inference.html#importing-data)
	+ [Analyzing satisfaction](pt_model_inference.html#analyzing-satisfaction)
	+ [Generate methods](pt_model_inference.html#generate-methods)








[Next

Tune Experiment run](pt_tune_experiment_run.html)
[Previous

Prompt Tuning](prompt_tuner.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)
















# Text from https://ibm.github.io/watsonx-ai-python-sdk/autoai.html








AutoAI - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](#)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





AutoAI[¶](#autoai "Link to this heading")
=========================================


This version of `ibm-watsonx-ai` client introduces support for AutoAI Experiments.



* [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
	+ [Configure optimizer with one data source](autoai_working_with_class_and_optimizer.html#configure-optimizer-with-one-data-source)
	+ [Configure optimizer for time series forecasting](autoai_working_with_class_and_optimizer.html#configure-optimizer-for-time-series-forecasting)
	+ [Configure optimizer for time series forecasting with supporting features](autoai_working_with_class_and_optimizer.html#configure-optimizer-for-time-series-forecasting-with-supporting-features)
	+ [Get configuration parameters](autoai_working_with_class_and_optimizer.html#get-configuration-parameters)
	+ [Fit optimizer](autoai_working_with_class_and_optimizer.html#fit-optimizer)
	+ [Get the run status and run details](autoai_working_with_class_and_optimizer.html#get-the-run-status-and-run-details)
	+ [Get data connections](autoai_working_with_class_and_optimizer.html#get-data-connections)
	+ [Pipeline summary](autoai_working_with_class_and_optimizer.html#pipeline-summary)
	+ [Get pipeline details](autoai_working_with_class_and_optimizer.html#get-pipeline-details)
	+ [Get pipeline](autoai_working_with_class_and_optimizer.html#get-pipeline)
	+ [Working with deployments](autoai_working_with_class_and_optimizer.html#working-with-deployments)
	+ [Web Service](autoai_working_with_class_and_optimizer.html#web-service)
	+ [Batch](autoai_working_with_class_and_optimizer.html#batch)
* [AutoAI experiment](autoai_experiment.html)
	+ [AutoAI](autoai_experiment.html#autoai)
		- [`AutoAI`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI)
			* [`AutoAI.ClassificationAlgorithms`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms)
				+ [`AutoAI.ClassificationAlgorithms.DT`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.DT)
				+ [`AutoAI.ClassificationAlgorithms.EX_TREES`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.EX_TREES)
				+ [`AutoAI.ClassificationAlgorithms.GB`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.GB)
				+ [`AutoAI.ClassificationAlgorithms.LGBM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.LGBM)
				+ [`AutoAI.ClassificationAlgorithms.LR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.LR)
				+ [`AutoAI.ClassificationAlgorithms.RF`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.RF)
				+ [`AutoAI.ClassificationAlgorithms.SnapBM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapBM)
				+ [`AutoAI.ClassificationAlgorithms.SnapDT`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapDT)
				+ [`AutoAI.ClassificationAlgorithms.SnapLR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapLR)
				+ [`AutoAI.ClassificationAlgorithms.SnapRF`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapRF)
				+ [`AutoAI.ClassificationAlgorithms.SnapSVM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapSVM)
				+ [`AutoAI.ClassificationAlgorithms.XGB`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.XGB)
			* [`AutoAI.DataConnectionTypes`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes)
				+ [`AutoAI.DataConnectionTypes.CA`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.CA)
				+ [`AutoAI.DataConnectionTypes.CN`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.CN)
				+ [`AutoAI.DataConnectionTypes.DS`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.DS)
				+ [`AutoAI.DataConnectionTypes.FS`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.FS)
				+ [`AutoAI.DataConnectionTypes.S3`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.S3)
			* [`AutoAI.ForecastingAlgorithms`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms)
				+ [`AutoAI.ForecastingAlgorithms.ARIMA`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.ARIMA)
				+ [`AutoAI.ForecastingAlgorithms.BATS`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.BATS)
				+ [`AutoAI.ForecastingAlgorithms.ENSEMBLER`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.ENSEMBLER)
				+ [`AutoAI.ForecastingAlgorithms.HW`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.HW)
				+ [`AutoAI.ForecastingAlgorithms.LR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.LR)
				+ [`AutoAI.ForecastingAlgorithms.RF`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.RF)
				+ [`AutoAI.ForecastingAlgorithms.SVM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.SVM)
			* [`AutoAI.Metrics`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics)
				+ [`AutoAI.Metrics.ACCURACY_AND_DISPARATE_IMPACT_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ACCURACY_AND_DISPARATE_IMPACT_SCORE)
				+ [`AutoAI.Metrics.ACCURACY_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ACCURACY_SCORE)
				+ [`AutoAI.Metrics.AVERAGE_PRECISION_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.AVERAGE_PRECISION_SCORE)
				+ [`AutoAI.Metrics.EXPLAINED_VARIANCE_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.EXPLAINED_VARIANCE_SCORE)
				+ [`AutoAI.Metrics.F1_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE)
				+ [`AutoAI.Metrics.F1_SCORE_MACRO`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE_MACRO)
				+ [`AutoAI.Metrics.F1_SCORE_MICRO`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE_MICRO)
				+ [`AutoAI.Metrics.F1_SCORE_WEIGHTED`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE_WEIGHTED)
				+ [`AutoAI.Metrics.LOG_LOSS`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.LOG_LOSS)
				+ [`AutoAI.Metrics.MEAN_ABSOLUTE_ERROR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEAN_ABSOLUTE_ERROR)
				+ [`AutoAI.Metrics.MEAN_SQUARED_ERROR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEAN_SQUARED_ERROR)
				+ [`AutoAI.Metrics.MEAN_SQUARED_LOG_ERROR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEAN_SQUARED_LOG_ERROR)
				+ [`AutoAI.Metrics.MEDIAN_ABSOLUTE_ERROR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEDIAN_ABSOLUTE_ERROR)
				+ [`AutoAI.Metrics.PRECISION_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE)
				+ [`AutoAI.Metrics.PRECISION_SCORE_MACRO`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE_MACRO)
				+ [`AutoAI.Metrics.PRECISION_SCORE_MICRO`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE_MICRO)
				+ [`AutoAI.Metrics.PRECISION_SCORE_WEIGHTED`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE_WEIGHTED)
				+ [`AutoAI.Metrics.R2_AND_DISPARATE_IMPACT_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.R2_AND_DISPARATE_IMPACT_SCORE)
				+ [`AutoAI.Metrics.R2_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.R2_SCORE)
				+ [`AutoAI.Metrics.RECALL_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE)
				+ [`AutoAI.Metrics.RECALL_SCORE_MACRO`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE_MACRO)
				+ [`AutoAI.Metrics.RECALL_SCORE_MICRO`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE_MICRO)
				+ [`AutoAI.Metrics.RECALL_SCORE_WEIGHTED`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE_WEIGHTED)
				+ [`AutoAI.Metrics.ROC_AUC_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ROC_AUC_SCORE)
				+ [`AutoAI.Metrics.ROOT_MEAN_SQUARED_ERROR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ROOT_MEAN_SQUARED_ERROR)
				+ [`AutoAI.Metrics.ROOT_MEAN_SQUARED_LOG_ERROR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ROOT_MEAN_SQUARED_LOG_ERROR)
			* [`AutoAI.PipelineTypes`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PipelineTypes)
				+ [`AutoAI.PipelineTypes.LALE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PipelineTypes.LALE)
				+ [`AutoAI.PipelineTypes.SKLEARN`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PipelineTypes.SKLEARN)
			* [`AutoAI.PredictionType`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType)
				+ [`AutoAI.PredictionType.BINARY`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.BINARY)
				+ [`AutoAI.PredictionType.CLASSIFICATION`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.CLASSIFICATION)
				+ [`AutoAI.PredictionType.FORECASTING`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.FORECASTING)
				+ [`AutoAI.PredictionType.MULTICLASS`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.MULTICLASS)
				+ [`AutoAI.PredictionType.REGRESSION`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.REGRESSION)
				+ [`AutoAI.PredictionType.TIMESERIES_ANOMALY_PREDICTION`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.TIMESERIES_ANOMALY_PREDICTION)
			* [`AutoAI.RegressionAlgorithms`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms)
				+ [`AutoAI.RegressionAlgorithms.DT`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.DT)
				+ [`AutoAI.RegressionAlgorithms.EX_TREES`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.EX_TREES)
				+ [`AutoAI.RegressionAlgorithms.GB`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.GB)
				+ [`AutoAI.RegressionAlgorithms.LGBM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.LGBM)
				+ [`AutoAI.RegressionAlgorithms.LR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.LR)
				+ [`AutoAI.RegressionAlgorithms.RF`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.RF)
				+ [`AutoAI.RegressionAlgorithms.RIDGE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.RIDGE)
				+ [`AutoAI.RegressionAlgorithms.SnapBM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.SnapBM)
				+ [`AutoAI.RegressionAlgorithms.SnapDT`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.SnapDT)
				+ [`AutoAI.RegressionAlgorithms.SnapRF`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.SnapRF)
				+ [`AutoAI.RegressionAlgorithms.XGB`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.XGB)
			* [`AutoAI.SamplingTypes`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes)
				+ [`AutoAI.SamplingTypes.FIRST_VALUES`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.FIRST_VALUES)
				+ [`AutoAI.SamplingTypes.LAST_VALUES`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.LAST_VALUES)
				+ [`AutoAI.SamplingTypes.RANDOM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.RANDOM)
				+ [`AutoAI.SamplingTypes.STRATIFIED`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.STRATIFIED)
			* [`AutoAI.TShirtSize`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize)
				+ [`AutoAI.TShirtSize.L`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.L)
				+ [`AutoAI.TShirtSize.M`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.M)
				+ [`AutoAI.TShirtSize.S`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.S)
				+ [`AutoAI.TShirtSize.XL`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.XL)
			* [`AutoAI.Transformers`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers)
				+ [`AutoAI.Transformers.ABS`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.ABS)
				+ [`AutoAI.Transformers.CBRT`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.CBRT)
				+ [`AutoAI.Transformers.COS`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.COS)
				+ [`AutoAI.Transformers.CUBE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.CUBE)
				+ [`AutoAI.Transformers.DIFF`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.DIFF)
				+ [`AutoAI.Transformers.DIVIDE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.DIVIDE)
				+ [`AutoAI.Transformers.FEATUREAGGLOMERATION`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.FEATUREAGGLOMERATION)
				+ [`AutoAI.Transformers.ISOFORESTANOMALY`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.ISOFORESTANOMALY)
				+ [`AutoAI.Transformers.LOG`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.LOG)
				+ [`AutoAI.Transformers.MAX`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.MAX)
				+ [`AutoAI.Transformers.MINMAXSCALER`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.MINMAXSCALER)
				+ [`AutoAI.Transformers.NXOR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.NXOR)
				+ [`AutoAI.Transformers.PCA`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.PCA)
				+ [`AutoAI.Transformers.PRODUCT`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.PRODUCT)
				+ [`AutoAI.Transformers.ROUND`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.ROUND)
				+ [`AutoAI.Transformers.SIGMOID`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SIGMOID)
				+ [`AutoAI.Transformers.SIN`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SIN)
				+ [`AutoAI.Transformers.SQRT`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SQRT)
				+ [`AutoAI.Transformers.SQUARE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SQUARE)
				+ [`AutoAI.Transformers.STDSCALER`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.STDSCALER)
				+ [`AutoAI.Transformers.SUM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SUM)
				+ [`AutoAI.Transformers.TAN`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.TAN)
			* [`AutoAI.optimizer()`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.optimizer)
			* [`AutoAI.runs()`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.runs)
* [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Web Service](autoai_deployment_modules.html#web-service)
		- [`WebService`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.WebService)
			* [`WebService.create()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.WebService.create)
			* [`WebService.delete()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.WebService.delete)
			* [`WebService.get()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.WebService.get)
			* [`WebService.get_params()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.WebService.get_params)
			* [`WebService.list()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.WebService.list)
			* [`WebService.score()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.WebService.score)
	+ [Batch](autoai_deployment_modules.html#batch)
		- [`Batch`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch)
			* [`Batch.create()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.create)
			* [`Batch.delete()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.delete)
			* [`Batch.get()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.get)
			* [`Batch.get_job_id()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.get_job_id)
			* [`Batch.get_job_params()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.get_job_params)
			* [`Batch.get_job_result()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.get_job_result)
			* [`Batch.get_job_status()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.get_job_status)
			* [`Batch.get_params()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.get_params)
			* [`Batch.list()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.list)
			* [`Batch.list_jobs()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.list_jobs)
			* [`Batch.rerun_job()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.rerun_job)
			* [`Batch.run_job()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.run_job)
			* [`Batch.score()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.score)








[Next

Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
[Previous

DataConnection Modules](dataconnection_modules.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)
















# Text from https://ibm.github.io/watsonx-ai-python-sdk/samples.html








Samples - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](#)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Samples[¶](#samples "Link to this heading")
===========================================


To view sample notebooks for IBM watsonx.ai, refer to [Python sample notebooks](https://github.com/IBM/watson-machine-learning-samples/blob/master/README.md).


Most of the watsonx.ai notebooks are also available on the [Resource hub](https://dataplatform.cloud.ibm.com/samples?context=wx).







[Next

Migration from `ibm_watson_machine_learning`](migration.html)
[Previous

Custom models](fm_working_with_custom_models.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)
















# Text from https://ibm.github.io/watsonx-ai-python-sdk/fm_models.html#








Models - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](#)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Models[¶](#models "Link to this heading")
=========================================


The `Model` module is an extension of `ModelInference` with langchain support (option to get WatsonxLLM wrapper for watsonx foundation models).



Modules[¶](#modules "Link to this heading")
-------------------------------------------



* [Model](fm_model.html)
	+ [`Model`](fm_model.html#ibm_watsonx_ai.foundation_models.Model)
		- [`Model.generate()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.generate)
		- [`Model.generate_text()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.generate_text)
		- [`Model.generate_text_stream()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.generate_text_stream)
		- [`Model.get_details()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.get_details)
		- [`Model.to_langchain()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.to_langchain)
		- [`Model.tokenize()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.tokenize)
	+ [Enums](fm_model.html#enums)
		- [`GenTextParamsMetaNames`](fm_model.html#metanames.GenTextParamsMetaNames)
		- [`GenTextReturnOptMetaNames`](fm_model.html#metanames.GenTextReturnOptMetaNames)
		- [`DecodingMethods`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods)
			* [`DecodingMethods.GREEDY`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods.GREEDY)
			* [`DecodingMethods.SAMPLE`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods.SAMPLE)
		- [`TextModels`](fm_model.html#TextModels)
		- [`ModelTypes`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes)
* [ModelInference](fm_model_inference.html)
	+ [`ModelInference`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference)
		- [`ModelInference.generate()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate)
		- [`ModelInference.generate_text()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate_text)
		- [`ModelInference.generate_text_stream()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate_text_stream)
		- [`ModelInference.get_details()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.get_details)
		- [`ModelInference.get_identifying_params()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.get_identifying_params)
		- [`ModelInference.to_langchain()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.to_langchain)
		- [`ModelInference.tokenize()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.tokenize)
* [`ModelInference` for Deployments](fm_deployments.html)
	+ [Infer text with deployments](fm_deployments.html#infer-text-with-deployments)
	+ [Creating `ModelInference` instance](fm_deployments.html#creating-modelinference-instance)
	+ [Generate methods](fm_deployments.html#generate-methods)









[Next

Model](fm_model.html)
[Previous

Embeddings](fm_embeddings.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Models](#)
	+ [Modules](#modules)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/prompt_template_manager.html








Prompt Template Manager - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](#)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Prompt Template Manager[¶](#prompt-template-manager "Link to this heading")
===========================================================================




*class* ibm\_watsonx\_ai.foundation\_models.prompts.PromptTemplateManager(*credentials=None*, *\**, *project\_id=None*, *space\_id=None*, *verify=None*, *api\_client=None*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompts/prompt_template.html#PromptTemplateManager)[¶](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager "Link to this definition")
Bases: `WMLResource`


Instantiate the prompt template manager.



Parameters:
* **credentials** ([*Credentials*](base.html#credentials.Credentials "credentials.Credentials")) – Credentials to watsonx.ai instance.
* **project\_id** (*str*) – ID of project
* **space\_id** (*str*) – ID of project
* **verify** (*bool* *or* *str**,* *optional*) – user can pass as verify one of following:
- the path to a CA\_BUNDLE file
- the path of directory with certificates of trusted CAs
- True - default path to truststore will be taken
- False - no verification will be made





Note


One of these parameters is required: [‘project\_id ‘, ‘space\_id’]



**Example**



```
from ibm_watsonx_ai import Credentials

from ibm_watsonx_ai.foundation_models.prompts import PromptTemplate, PromptTemplateManager
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

prompt_mgr = PromptTemplateManager(
                credentials=Credentials(
                    api_key="***",
                    url="https://us-south.ml.cloud.ibm.com"
                ),
                project_id="*****"
                )

prompt_template = PromptTemplate(name="My prompt",
                                 model_id=ModelTypes.GRANITE_13B_CHAT_V2,
                                 input_prefix="Human:",
                                 output_prefix="Assistant:",
                                 input_text="What is {object} and how does it work?",
                                 input_variables=['object'],
                                 examples=[['What is the Stock Market?',
                                            'A stock market is a place where investors buy and sell shares of publicly traded companies.']])

stored_prompt_template = prompt_mgr.store_prompt(prompt_template)
print(stored_prompt_template.prompt_id)   # id of prompt template asset

```



Note


Here’s an example of how you can pass variables to your deployed prompt template.



```
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

meta_props = {
    client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
    client.deployments.ConfigurationMetaNames.ONLINE: {},
    client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: ModelTypes.GRANITE_13B_CHAT_V2
    }

deployment_details = client.deployments.create(stored_prompt_template.prompt_id, meta_props)

client.deployments.generate_text(
    deployment_id=deployment_details["metadata"]["id"],
    params={
        GenTextParamsMetaNames.PROMPT_VARIABLES: {
            "object": "brain"
        }
    }
)

```





delete\_prompt(*prompt\_id*, *\**, *force=False*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompts/prompt_template.html#PromptTemplateManager.delete_prompt)[¶](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.delete_prompt "Link to this definition")
Remove prompt template from project or space.



Parameters:
* **prompt\_id** (*str*) – Id of prompt template that will be delete.
* **force** (*bool*) – If True then prompt template is unlocked and then delete, defaults to False.



Returns:
Status ‘SUCCESS’ if the prompt template is successfully deleted.



Return type:
str




**Example**



```
prompt_mgr.delete_prompt(prompt_id)  # delete if asset is unlocked

```





get\_lock(*prompt\_id*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompts/prompt_template.html#PromptTemplateManager.get_lock)[¶](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.get_lock "Link to this definition")
Get the current locked state of a prompt template.



Parameters:
**prompt\_id** (*str*) – Id of prompt template



Returns:
Information about locked state of prompt template asset.



Return type:
dict




**Example**



```
print(prompt_mgr.get_lock(prompt_id))

```





list(*\**, *limit=None*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompts/prompt_template.html#PromptTemplateManager.list)[¶](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.list "Link to this definition")
List all available prompt templates in the DataFrame format.



Parameters:
**limit** (*int**,* *optional*) – limit number of fetched records, defaults to None.



Returns:
DataFrame of fundamental properties of available prompts.



Return type:
pandas.core.frame.DataFrame




**Example**



```
prompt_mgr.list(limit=5)    # list of 5 recent created prompt template assets

```



Hint


Additionally you can sort available prompt templates by “LAST MODIFIED” field.



```
df_prompts = prompt_mgr.list()
df_prompts.sort_values("LAST MODIFIED", ascending=False)

```






load\_prompt(*prompt\_id*, *astype=PromptTemplateFormats.PROMPTTEMPLATE*, *\**, *prompt\_variables=None*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompts/prompt_template.html#PromptTemplateManager.load_prompt)[¶](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.load_prompt "Link to this definition")
Retrieve a prompt template asset.



Parameters:
* **prompt\_id** (*str*) – Id of prompt template which is processed.
* **astype** ([*PromptTemplateFormats*](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats "ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats")) – Type of return object.
* **prompt\_variables** (*dict**[**str**,* *str**]*) – dictionary of input variables and values with which input variables will be replaced.



Returns:
Prompt template asset.



Return type:
[FreeformPromptTemplate](#ibm_watsonx_ai.foundation_models.prompts.FreeformPromptTemplate "ibm_watsonx_ai.foundation_models.prompts.FreeformPromptTemplate") | [PromptTemplate](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplate "ibm_watsonx_ai.foundation_models.prompts.PromptTemplate") | [DetachedPromptTemplate](#ibm_watsonx_ai.foundation_models.prompts.DetachedPromptTemplate "ibm_watsonx_ai.foundation_models.prompts.DetachedPromptTemplate") | str | langchain.prompts.PromptTemplate




**Example**



```
loaded_prompt_template = prompt_mgr.load_prompt(prompt_id)
loaded_prompt_template_lc = prompt_mgr.load_prompt(prompt_id, PromptTemplateFormats.LANGCHAIN)
loaded_prompt_template_string = prompt_mgr.load_prompt(prompt_id, PromptTemplateFormats.STRING)

```





lock(*prompt\_id*, *force=False*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompts/prompt_template.html#PromptTemplateManager.lock)[¶](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.lock "Link to this definition")
Lock the prompt template if it is unlocked and user has permission to do that.



Parameters:
* **prompt\_id** (*str*) – Id of prompt template.
* **force** (*bool*) – If True, method forcefully overwrite a lock.



Returns:
Status ‘SUCCESS’ or response content after an attempt to lock prompt template.



Return type:
dict




**Example**



```
prompt_mgr.lock(prompt_id)

```





store\_prompt(*prompt\_template*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompts/prompt_template.html#PromptTemplateManager.store_prompt)[¶](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.store_prompt "Link to this definition")
Store a new prompt template.



Parameters:
**prompt\_template** (*(*[*FreeformPromptTemplate*](#ibm_watsonx_ai.foundation_models.prompts.FreeformPromptTemplate "ibm_watsonx_ai.foundation_models.prompts.FreeformPromptTemplate") *|* [*PromptTemplate*](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplate "ibm_watsonx_ai.foundation_models.prompts.PromptTemplate") *|* [*DetachedPromptTemplate*](#ibm_watsonx_ai.foundation_models.prompts.DetachedPromptTemplate "ibm_watsonx_ai.foundation_models.prompts.DetachedPromptTemplate") *|* *langchain.prompts.PromptTemplate**)*) – PromptTemplate to be stored.



Returns:
PromptTemplate object initialized with values provided in the server response object.



Return type:
[FreeformPromptTemplate](#ibm_watsonx_ai.foundation_models.prompts.FreeformPromptTemplate "ibm_watsonx_ai.foundation_models.prompts.FreeformPromptTemplate") | [PromptTemplate](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplate "ibm_watsonx_ai.foundation_models.prompts.PromptTemplate") | [DetachedPromptTemplate](#ibm_watsonx_ai.foundation_models.prompts.DetachedPromptTemplate "ibm_watsonx_ai.foundation_models.prompts.DetachedPromptTemplate")







unlock(*prompt\_id*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompts/prompt_template.html#PromptTemplateManager.unlock)[¶](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.unlock "Link to this definition")
Unlock the prompt template if it is locked and user has permission to do that.



Parameters:
**prompt\_id** (*str*) – Id of prompt template.



Returns:
Response content after an attempt to unlock prompt template.



Return type:
dict




**Example**



```
prompt_mgr.unlock(prompt_id)

```





update\_prompt(*prompt\_id*, *prompt\_template*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompts/prompt_template.html#PromptTemplateManager.update_prompt)[¶](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.update_prompt "Link to this definition")
Update prompt template data.



Parameters:
* **prompt\_id** (*str*) – Id of the updated prompt template.
* **prompt** ([*FreeformPromptTemplate*](#ibm_watsonx_ai.foundation_models.prompts.FreeformPromptTemplate "ibm_watsonx_ai.foundation_models.prompts.FreeformPromptTemplate") *|* [*PromptTemplate*](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplate "ibm_watsonx_ai.foundation_models.prompts.PromptTemplate") *|* [*DetachedPromptTemplate*](#ibm_watsonx_ai.foundation_models.prompts.DetachedPromptTemplate "ibm_watsonx_ai.foundation_models.prompts.DetachedPromptTemplate")) – PromptTemplate with new data.



Returns:
metadata of updated deployment



Return type:
dict




**Example**



```
updataed_prompt_template = PromptTemplate(name="New name")
prompt_mgr.update_prompt(prompt_id, prompt_template)  # {'name': 'New name'} in metadata

```






*class* ibm\_watsonx\_ai.foundation\_models.prompts.PromptTemplate(*name=None*, *model\_id=None*, *model\_params=None*, *template\_version=None*, *task\_ids=None*, *description=None*, *input\_text=None*, *input\_variables=None*, *instruction=None*, *input\_prefix=None*, *output\_prefix=None*, *examples=None*, *validate\_template=True*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompts/prompt_template.html#PromptTemplate)[¶](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplate "Link to this definition")
Bases: `BasePromptTemplate`


Parameter storage for a structured Prompt Template.



Parameters:
* **prompt\_id** (*str**,* *attribute setting not allowed*) – Id of prompt template, defaults to None.
* **created\_at** (*str**,* *attribute setting not allowed*) – Time the prompt was created (UTC), defaults to None.
* **lock** (*PromptTemplateLock* *|* *None**,* *attribute setting not allowed*) – Locked state of asset, defaults to None.
* **is\_template** (*bool* *|* *None**,* *attribute setting not allowed*) – True if prompt is a template, False otherwise; defaults to None.
* **name** (*str**,* *optional*) – Prompt template name, defaults to None.
* **model\_id** ([*ModelTypes*](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes "ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes") *|* *str* *|* *None**,* *optional*) – Foundation model id, defaults to None.
* **model\_params** (*dict**,* *optional*) – Model parameters, defaults to None.
* **template\_version** (*str**,* *optional*) – Semantic version for tracking in IBM AI Factsheets, defaults to None.
* **task\_ids** (*list**[**str**]* *|* *None**,* *optional*) – List of task ids, defaults to None.
* **description** (*str**,* *optional*) – Prompt template asset description, defaults to None.
* **input\_text** (*str**,* *optional*) – Input text for prompt, defaults to None.
* **input\_variables** (*(**list* *|* *dict**[**str**,* *dict**[**str**,* *str**]**]**)**,* *optional*) – Input variables can be present in fields: instruction,
input\_prefix, output\_prefix, input\_text, examples
and are identified by braces (‘{’ and ‘}’), defaults to None.
* **instruction** (*str**,* *optional*) – Instruction for model, defaults to None.
* **input\_prefix** (*str**,* *optional*) – Prefix string placed before input text, defaults to None.
* **output\_prefix** (*str**,* *optional*) – Prefix before model response, defaults to None.
* **examples** (*list**[**list**[**str**]**]**]**,* *optional*) – Examples may help the model to adjust the response; [[input1, output1], …], defaults to None.
* **validate\_template** (*bool**,* *optional*) – If True, the Prompt Template is validated for the presence of input variables, defaults to True.



Raises:
**ValidationError** – If the set of input\_variables is not consistent with the input variables present in the template.
Raises only when validate\_template is set to True.




**Examples**


Example of invalid Prompt Template:



```
prompt_template = PromptTemplate(
    name="My structured prompt",
    model_id="ibm/granite-13b-chat-v2"
    input_text='What are the most famous monuments in ?',
    input_variables=['country'])

# Traceback (most recent call last):
#     ...
# ValidationError: Invalid prompt template; check for mismatched or missing input variables. Missing input variable: {'country'}

```


Example of the valid Prompt Template:



```
prompt_template = PromptTemplate(
    name="My structured prompt",
    model_id="ibm/granite-13b-chat-v2"
    input_text='What are the most famous monuments in {country}?',
    input_variables=['country'])

```





*class* ibm\_watsonx\_ai.foundation\_models.prompts.FreeformPromptTemplate(*name=None*, *model\_id=None*, *model\_params=None*, *template\_version=None*, *task\_ids=None*, *description=None*, *input\_text=None*, *input\_variables=None*, *validate\_template=True*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompts/prompt_template.html#FreeformPromptTemplate)[¶](#ibm_watsonx_ai.foundation_models.prompts.FreeformPromptTemplate "Link to this definition")
Bases: `BasePromptTemplate`


Storage for free form Prompt Template asset parameters.



Parameters:
* **prompt\_id** (*str**,* *attribute setting not allowed*) – Id of prompt template, defaults to None.
* **created\_at** (*str**,* *attribute setting not allowed*) – Time the prompt was created (UTC), defaults to None.
* **lock** (*PromptTemplateLock* *|* *None**,* *attribute setting not allowed*) – Locked state of asset, defaults to None.
* **is\_template** (*bool* *|* *None**,* *attribute setting not allowed*) – True if prompt is a template, False otherwise; defaults to None.
* **name** (*str**,* *optional*) – Prompt template name, defaults to None.
* **model\_id** ([*ModelTypes*](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes "ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes") *|* *str* *|* *None**,* *optional*) – Foundation model id, defaults to None.
* **model\_params** (*dict**,* *optional*) – Model parameters, defaults to None.
* **template\_version** (*str**,* *optional*) – Semantic version for tracking in IBM AI Factsheets, defaults to None.
* **task\_ids** (*list**[**str**]* *|* *None**,* *optional*) – List of task ids, defaults to None.
* **description** (*str**,* *optional*) – Prompt template asset description, defaults to None.
* **input\_text** (*str**,* *optional*) – Input text for prompt, defaults to None.
* **input\_variables** (*(**list* *|* *dict**[**str**,* *dict**[**str**,* *str**]**]**)**,* *optional*) – Input variables can be present in field input\_text
and are identified by braces (‘{’ and ‘}’), defaults to None.
* **validate\_template** (*bool**,* *optional*) – If True, the Prompt Template is validated for the presence of input variables, defaults to True.



Raises:
**ValidationError** – If the set of input\_variables is not consistent with the input variables present in the template.
Raises only when validate\_template is set to True.




**Examples**


Example of invalid Freeform Prompt Template:



```
prompt_template = FreeformPromptTemplate(
    name="My freeform prompt",
    model_id="ibm/granite-13b-chat-v2",
    input_text='What are the most famous monuments in ?',
    input_variables=['country'])

# Traceback (most recent call last):
#    ...
# ValidationError: Invalid prompt template; check for mismatched or missing input variables. Missing input variable: {'country'}

```


Example of the valid Freeform Prompt Template



```
prompt_template = FreeformPromptTemplate(
    name="My freeform prompt",
    model_id="ibm/granite-13b-chat-v2"
    input_text='What are the most famous monuments in {country}?',
    input_variables=['country'])

```





*class* ibm\_watsonx\_ai.foundation\_models.prompts.DetachedPromptTemplate(*name=None*, *model\_id=None*, *model\_params=None*, *template\_version=None*, *task\_ids=None*, *description=None*, *input\_text=None*, *input\_variables=None*, *detached\_prompt\_id=None*, *detached\_model\_id=None*, *detached\_model\_provider=None*, *detached\_prompt\_url=None*, *detached\_prompt\_additional\_information=None*, *detached\_model\_name=None*, *detached\_model\_url=None*, *validate\_template=True*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompts/prompt_template.html#DetachedPromptTemplate)[¶](#ibm_watsonx_ai.foundation_models.prompts.DetachedPromptTemplate "Link to this definition")
Bases: `BasePromptTemplate`


Storage for detached Prompt Template parameters.



Parameters:
* **prompt\_id** (*str**,* *attribute setting not allowed*) – Id of prompt template, defaults to None.
* **created\_at** (*str**,* *attribute setting not allowed*) – Time the prompt was created (UTC), defaults to None.
* **lock** (*PromptTemplateLock* *|* *None**,* *attribute setting not allowed*) – Locked state of asset, defaults to None.
* **is\_template** (*bool* *|* *None**,* *attribute setting not allowed*) – True if prompt is a template, False otherwise; defaults to None.
* **name** (*str**,* *optional*) – Prompt template name, defaults to None.
* **model\_id** ([*ModelTypes*](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes "ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes") *|* *str* *|* *None**,* *optional*) – Foundation model id, defaults to None.
* **model\_params** (*dict**,* *optional*) – Model parameters, defaults to None.
* **template\_version** (*str**,* *optional*) – Semantic version for tracking in IBM AI Factsheets, defaults to None.
* **task\_ids** (*list**[**str**]* *|* *None**,* *optional*) – List of task ids, defaults to None.
* **description** (*str**,* *optional*) – Prompt template asset description, defaults to None.
* **input\_text** (*str**,* *optional*) – Input text for prompt, defaults to None.
* **input\_variables** (*(**list* *|* *dict**[**str**,* *dict**[**str**,* *str**]**]**)**,* *optional*) – Input variables can be present in field: input\_text
and are identified by braces (‘{’ and ‘}’), defaults to None.
* **detached\_prompt\_id** (*str* *|* *None**,* *optional*) – Id of the external prompt, defaults to None
* **detached\_model\_id** (*str* *|* *None**,* *optional*) – Id of the external model, defaults to None
* **detached\_model\_provider** (*str* *|* *None**,* *optional*) – External model provider, defaults to None
* **detached\_prompt\_url** (*str* *|* *None**,* *optional*) – URL for the external prompt, defaults to None
* **detached\_prompt\_additional\_information** (*list**[**dict**[**str**,* *Any**]**]* *|* *None**,* *optional*) – Additional information of the external prompt, defaults to None
* **detached\_model\_name** (*str* *|* *None**,* *optional*) – Name of the external model, defaults to None
* **detached\_model\_url** (*str* *|* *None**,* *optional*) – URL for the external model, defaults to None
* **validate\_template** (*bool**,* *optional*) – If True, the Prompt Template is validated for the presence of input variables, defaults to True.



Raises:
**ValidationError** – If the set of input\_variables is not consistent with the input variables present in the template.
Raises only when validate\_template is set to True.




**Examples**


Example of invalid Detached Prompt Template:



```
prompt_template = DetachedPromptTemplate(
    name="My detached prompt",
    model_id="<some model>",
    input_text='What are the most famous monuments in ?',
    input_variables=['country'],
    detached_prompt_id="<prompt id>",
    detached_model_id="<model id>",
    detached_model_provider="<provider>",
    detached_prompt_url="<url>",
    detached_prompt_additional_information=[[{"key":"value"}]]},
    detached_model_name="<model name>",
    detached_model_url ="<model url>")

# Traceback (most recent call last):
#     ...
# ValidationError: Invalid prompt template; check for mismatched or missing input variables. Missing input variable: {'country'}

```


Example of the valid Detached Prompt Template:



```
prompt_template = DetachedPromptTemplate(
    name="My detached prompt",
    model_id="<some model>",
    input_text='What are the most famous monuments in {country}?',
    input_variables=['country'],
    detached_prompt_id="<prompt id>",
    detached_model_id="<model id>",
    detached_model_provider="<provider>",
    detached_prompt_url="<url>",
    detached_prompt_additional_information=[[{"key":"value"}]]},
    detached_model_name="<model name>",
    detached_model_url ="<model url>"))

```




Enums[¶](#enums "Link to this heading")
---------------------------------------




*class* ibm\_watsonx\_ai.foundation\_models.utils.enums.PromptTemplateFormats(*value*)[[source]](_modules/ibm_watsonx_ai/foundation_models/utils/enums.html#PromptTemplateFormats)[¶](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats "Link to this definition")
Bases: `Enum`


Supported formats of loaded prompt template.




LANGCHAIN *= 'langchain'*[¶](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats.LANGCHAIN "Link to this definition")



PROMPTTEMPLATE *= 'prompt'*[¶](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats.PROMPTTEMPLATE "Link to this definition")



STRING *= 'string'*[¶](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats.STRING "Link to this definition")








[Next

Extensions](fm_extensions.html)
[Previous

Tune Experiment](tune_experiment.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Prompt Template Manager](#)
	+ [`PromptTemplateManager`](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager)
		- [`PromptTemplateManager.delete_prompt()`](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.delete_prompt)
		- [`PromptTemplateManager.get_lock()`](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.get_lock)
		- [`PromptTemplateManager.list()`](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.list)
		- [`PromptTemplateManager.load_prompt()`](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.load_prompt)
		- [`PromptTemplateManager.lock()`](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.lock)
		- [`PromptTemplateManager.store_prompt()`](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.store_prompt)
		- [`PromptTemplateManager.unlock()`](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.unlock)
		- [`PromptTemplateManager.update_prompt()`](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.update_prompt)
	+ [`PromptTemplate`](#ibm_watsonx_ai.foundation_models.prompts.PromptTemplate)
	+ [`FreeformPromptTemplate`](#ibm_watsonx_ai.foundation_models.prompts.FreeformPromptTemplate)
	+ [`DetachedPromptTemplate`](#ibm_watsonx_ai.foundation_models.prompts.DetachedPromptTemplate)
	+ [Enums](#enums)
		- [`PromptTemplateFormats`](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats)
			* [`PromptTemplateFormats.LANGCHAIN`](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats.LANGCHAIN)
			* [`PromptTemplateFormats.PROMPTTEMPLATE`](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats.PROMPTTEMPLATE)
			* [`PromptTemplateFormats.STRING`](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats.STRING)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/changelog.html








Changelog - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](#)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Changelog[¶](#changelog "Link to this heading")
===============================================



1.0.10[¶](#id1 "Link to this heading")
--------------------------------------


Features:


* Added support for different input modes of Prompt Templates(`PromptTemplate`, `FreeformPromptTemplate`, `DetachedPromptTemplate`).


Bug Fixes:


* Fixed validation for tech preview models.
* Fixed set project for git-based-project in new scenario.
* Fixed used filters for `get_model_specs` method.




1.0.9[¶](#id2 "Link to this heading")
-------------------------------------


Features:


* Added support for tech preview models
* Added additional information in request headers.
* Extend VectorStore functionalities via concrete wrappers(For Vector Index notebooks)




1.0.8[¶](#id3 "Link to this heading")
-------------------------------------


Features:


* Added `validate_prompt_variables` parameter to generate method in `Model` and `ModelInference` class.
* Added `hardware_spec` support in Batch class.


Bug Fixes:


* Fixed correct schema for timeseries-anomaly-prediction prediction type.




1.0.6[¶](#id4 "Link to this heading")
-------------------------------------


Bug Fixes:


* Added more clear Error message when user pass invalid credentials.
* Fixed “Invalid authorization token” error when initializing the client with the “USER\_ACCESS\_TOKEN” environment variable on a cluster




1.0.5[¶](#id5 "Link to this heading")
-------------------------------------


Features:


* Added auto-generated Enum classes (TextModels, EmbeddingModels, PromptTunableModels) with available models


Bug Fixes:


* Better filtering of supported runtimes for r-scripts
* Fixed downloading model content to file
* Improved scaling Prompt Tuning charts
* Provided a clearer error message when a user passes an incorrect URL to the cluster




1.0.4[¶](#id6 "Link to this heading")
-------------------------------------


Features:


* Added forecast\_window parameter support for online deployment scoring




1.0.3[¶](#id7 "Link to this heading")
-------------------------------------


Features:


* Milvus support for RAG extension
* Autogenerated changelog
* Travis is tagging the version during push into production


Bug Fixes:


* Reading data assets as binary when flight is unavailable
* next resource generator type fixed, other internal type issues fixed
* Reading tabular dataset with non-unique columns
* Deprecation warnings removed when using fm\_model\_inference




1.0.2[¶](#id8 "Link to this heading")
-------------------------------------


Features:


* Added get and get\_item methods for better compatibility with SDK v0


Bug Fixes:


* Relaxed package version checking
* Fixed AutoAI initialization without version in credentials
* Fixed calls to wx endpoints in git base project
* Fixed backward compatibility of WebService and Batch class for credentials in dictionary




1.0.1[¶](#id9 "Link to this heading")
-------------------------------------


Bug Fixes:


* Hotfix for imports




1.0.0[¶](#id10 "Link to this heading")
--------------------------------------


Features:


* RAGutils module added
* Getting foundation models specs moved under foundation models object
* Credentials as object + proxies supported
* WCA service support


Bug Fixes:


* Minor corrections/improvements to Foundation Models module








[Previous

V1 Migration Guide](migration_v1.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Changelog](#)
	+ [1.0.10](#id1)
	+ [1.0.9](#id2)
	+ [1.0.8](#id3)
	+ [1.0.6](#id4)
	+ [1.0.5](#id5)
	+ [1.0.4](#id6)
	+ [1.0.3](#id7)
	+ [1.0.2](#id8)
	+ [1.0.1](#id9)
	+ [1.0.0](#id10)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/fm_deployments.html








ModelInference for Deployments - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](#)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





`ModelInference` for Deployments[¶](#modelinference-for-deployments "Link to this heading")
===========================================================================================


This section shows how to use ModelInference module with created deployment.


There are two ways to infer text using the [deployments](pt_model_inference.html#generate-text-deployments) module or using [ModelInference](pt_model_inference.html#generate-text-modelinference) module .



Infer text with deployments[¶](#infer-text-with-deployments "Link to this heading")
-----------------------------------------------------------------------------------


You can directly query `generate_text` using the deployments module.



```
client.deployments.generate_text(
    prompt="Example prompt",
    deployment_id=deployment_id)

```




Creating `ModelInference` instance[¶](#creating-modelinference-instance "Link to this heading")
-----------------------------------------------------------------------------------------------


At the beginning, it is recommended to define parameters (later used by module).



```
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

generate_params = {
    GenParams.MAX_NEW_TOKENS: 25,
    GenParams.STOP_SEQUENCES: ["\n"]
}

```


Create the ModelInference itself, using credentials and `project_id` / `space_id` or the previously initialized APIClient (see [APIClient initialization](pt_model_inference.html#api-client-init)).



```
from ibm_watsonx_ai.foundation_models import ModelInference

deployed_model = ModelInference(
    deployment_id=deployment_id,
    params=generate_params,
    credentials=credentials,
    project_id=project_id
)

# OR

deployed_model = ModelInference(
    deployment_id=deployment_id,
    params=generate_params,
    api_client=client
)

```


You can directly query `generate_text` using the `ModelInference` object.



```
deployed_model.generate_text(prompt="Example prompt")

```




Generate methods[¶](#generate-methods "Link to this heading")
-------------------------------------------------------------


The detailed explanation of available generate methods with exact parameters can be found in the [ModelInferece class](fm_model_inference.html#model-inference-class).


With previously created `deployed_model` object, it is possible to generate a text stream (generator) using defined inference and `generate_text_stream()` method.



```
for token in deployed_model.generate_text_stream(prompt=input_prompt):
    print(token, end="")
'$10 Powerchill Leggings'

```


And also receive more detailed result with `generate()`.



```
details = deployed_model.generate(prompt=input_prompt, params=gen_params)
print(details)
{
    'model_id': 'google/flan-t5-xl',
    'created_at': '2023-11-17T15:32:57.401Z',
    'results': [
        {
        'generated_text': '$10 Powerchill Leggings',
        'generated_token_count': 8,
        'input_token_count': 73,
        'stop_reason': 'eos_token'
        }
    ],
    'system': {'warnings': []}
}

```








[Next

Prompt Tuning](prompt_tuner.html)
[Previous

ModelInference](fm_model_inference.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [`ModelInference` for Deployments](#)
	+ [Infer text with deployments](#infer-text-with-deployments)
	+ [Creating `ModelInference` instance](#creating-modelinference-instance)
	+ [Generate methods](#generate-methods)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/tune_experiment.html








Tune Experiment - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](#)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Tune Experiment[¶](#tune-experiment "Link to this heading")
===========================================================



TuneExperiment[¶](#tuneexperiment "Link to this heading")
---------------------------------------------------------




*class* ibm\_watsonx\_ai.experiment.fm\_tune.TuneExperiment(*credentials*, *project\_id=None*, *space\_id=None*, *verify=None*)[[source]](_modules/ibm_watsonx_ai/experiment/fm_tune/tune_experiment.html#TuneExperiment)[¶](#ibm_watsonx_ai.experiment.fm_tune.TuneExperiment "Link to this definition")
Bases: `BaseExperiment`


TuneExperiment class for tuning models with prompts.



Parameters:
* **credentials** (*ibm\_watsonx\_ai.Credentials* *or* *dict*) – credentials to Watson Machine Learning instance
* **project\_id** (*str**,* *optional*) – ID of the Watson Studio project
* **space\_id** (*str**,* *optional*) – ID of the Watson Studio Space
* **verify** (*bool* *or* *str**,* *optional*) – user can pass as verify one of following:



	+ the path to a CA\_BUNDLE file
	+ the path of directory with certificates of trusted CAs
	+ True - default path to truststore will be taken
	+ False - no verification will be made




**Example**



```
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.experiment import TuneExperiment

experiment = TuneExperiment(
    credentials=Credentials(...),
    project_id="...",
    space_id="...")

```




prompt\_tuner(*name*, *task\_id*, *description=None*, *base\_model=None*, *accumulate\_steps=None*, *batch\_size=None*, *init\_method=None*, *init\_text=None*, *learning\_rate=None*, *max\_input\_tokens=None*, *max\_output\_tokens=None*, *num\_epochs=None*, *verbalizer=None*, *tuning\_type=None*, *auto\_update\_model=True*, *group\_by\_name=False*)[[source]](_modules/ibm_watsonx_ai/experiment/fm_tune/tune_experiment.html#TuneExperiment.prompt_tuner)[¶](#ibm_watsonx_ai.experiment.fm_tune.TuneExperiment.prompt_tuner "Link to this definition")
Initialize a PromptTuner module.



Parameters:
* **name** (*str*) – name for the PromptTuner
* **task\_id** (*str*) – task that is targeted for this model. Example: experiment.Tasks.CLASSIFICATION


Possible values:



	+ experiment.Tasks.CLASSIFICATION: ‘classification’ (default)
	+ experiment.Tasks.QUESTION\_ANSWERING: ‘question\_answering’
	+ experiment.Tasks.SUMMARIZATION: ‘summarization’
	+ experiment.Tasks.RETRIEVAL\_AUGMENTED\_GENERATION: ‘retrieval\_augmented\_generation’
	+ experiment.Tasks.GENERATION: ‘generation’
	+ experiment.Tasks.CODE\_GENERATION\_AND\_CONVERSION: ‘code’
	+ experiment.Tasks.EXTRACTION: ‘extraction
* **description** (*str**,* *optional*) – description
* **base\_model** (*str**,* *optional*) – model id of the base model for this prompt tuning. Example: google/flan-t5-xl
* **accumulate\_steps** (*int**,* *optional*) – Number of steps to be used for gradient accumulation. Gradient accumulation
refers to a method of collecting gradient for configured number of steps instead of updating
the model variables at every step and then applying the update to model variables.
This can be used as a tool to overcome smaller batch size limitation.
Often also referred in conjunction with “effective batch size”. Possible values: 1 ≤ value ≤ 128,
default value: 16
* **batch\_size** (*int**,* *optional*) – The batch size is a number of samples processed before the model is updated.
Possible values: 1 ≤ value ≤ 16, default value: 16
* **init\_method** (*str**,* *optional*) – text method requires init\_text to be set. Allowable values: [random, text],
default value: random
* **init\_text** (*str**,* *optional*) – initialization text to be used if init\_method is set to text otherwise this will be ignored.
* **learning\_rate** (*float**,* *optional*) – learning rate to be used while tuning prompt vectors. Possible values: 0.01 ≤ value ≤ 0.5,
default value: 0.3
* **max\_input\_tokens** (*int**,* *optional*) – maximum length of input tokens being considered. Possible values: 1 ≤ value ≤ 256,
default value: 256
* **max\_output\_tokens** (*int**,* *optional*) – maximum length of output tokens being predicted. Possible values: 1 ≤ value ≤ 128
default value: 128
* **num\_epochs** (*int**,* *optional*) – number of epochs to tune the prompt vectors, this affects the quality of the trained model.
Possible values: 1 ≤ value ≤ 50, default value: 20
* **verbalizer** (*str**,* *optional*) – verbalizer template to be used for formatting data at train and inference time.
This template may use brackets to indicate where fields from the data model must be rendered.
The default value is “{{input}}” which means use the raw text, default value: Input: {{input}} Output:
* **tuning\_type** (*str**,* *optional*) – type of Peft (Parameter-Efficient Fine-Tuning) config to build.
Allowable values: [experiment.PromptTuningTypes.PT], default value: experiment.PromptTuningTypes.PT
* **auto\_update\_model** (*bool**,* *optional*) – define if model should be automatically updated, default value: True
* **group\_by\_name** (*bool**,* *optional*) – define if tunings should be grouped by name, default value: False




**Examples**



```
from ibm_watsonx_ai.experiment import TuneExperiment

experiment = TuneExperiment(...)

prompt_tuner = experiment.prompt_tuner(
    name="prompt tuning name",
    task_id=experiment.Tasks.CLASSIFICATION,
    base_model='google/flan-t5-xl',
    accumulate_steps=32,
    batch_size=16,
    learning_rate=0.2,
    max_input_tokens=256,
    max_output_tokens=2,
    num_epochs=6,
    tuning_type=experiment.PromptTuningTypes.PT,
    verbalizer="Extract the satisfaction from the comment. Return simple '1' for satisfied customer or '0' for unsatisfied. Input: {{input}} Output: ",
    auto_update_model=True)

```





runs(*\**, *filter*)[[source]](_modules/ibm_watsonx_ai/experiment/fm_tune/tune_experiment.html#TuneExperiment.runs)[¶](#ibm_watsonx_ai.experiment.fm_tune.TuneExperiment.runs "Link to this definition")
Get the historical tuning runs but with name filter.



Parameters:
**filter** (*str*) – filter, user can choose which runs to fetch specifying tuning name




**Examples**



```
from ibm_watsonx_ai.experiment import TuneExperiment

experiment = TuneExperiment(...)
experiment.runs(filter='prompt tuning name').list()

```






Tune Runs[¶](#tune-runs "Link to this heading")
-----------------------------------------------




*class* ibm\_watsonx\_ai.experiment.fm\_tune.TuneRuns(*client*, *filter=None*, *limit=50*)[[source]](_modules/ibm_watsonx_ai/experiment/fm_tune/tune_runs.html#TuneRuns)[¶](#ibm_watsonx_ai.experiment.fm_tune.TuneRuns "Link to this definition")
Bases: `object`


TuneRuns class is used to work with historical PromptTuner runs.



Parameters:
* **client** ([*APIClient*](base.html#client.APIClient "client.APIClient")) – APIClient to handle service operations
* **filter** (*str**,* *optional*) – filter, user can choose which runs to fetch specifying tuning name
* **limit** (*int*) – int number of records to be returned






get\_run\_details(*run\_id=None*, *include\_metrics=False*)[[source]](_modules/ibm_watsonx_ai/experiment/fm_tune/tune_runs.html#TuneRuns.get_run_details)[¶](#ibm_watsonx_ai.experiment.fm_tune.TuneRuns.get_run_details "Link to this definition")
Get run details. If run\_id is not supplied, last run will be taken.



Parameters:
* **run\_id** (*str**,* *optional*) – ID of the run
* **include\_metrics** (*bool**,* *optional*) – indicates to include metrics in the training details output



Returns:
run configuration parameters



Return type:
dict




**Example**



```
from ibm_watsonx_ai.experiment import TuneExperiment
experiment = TuneExperiment(credentials, ...)

experiment.runs.get_run_details(run_id='02bab973-ae83-4283-9d73-87b9fd462d35')
experiment.runs.get_run_details()

```





get\_tuner(*run\_id*)[[source]](_modules/ibm_watsonx_ai/experiment/fm_tune/tune_runs.html#TuneRuns.get_tuner)[¶](#ibm_watsonx_ai.experiment.fm_tune.TuneRuns.get_tuner "Link to this definition")
Create instance of PromptTuner based on tuning run with specific run\_id.



Parameters:
**run\_id** (*str*) – ID of the run



Returns:
prompt tuner object



Return type:
PromptTuner class instance




**Example**



```
from ibm_watsonx_ai.experiment import TuneExperiment

experiment = TuneExperiment(credentials, ...)
historical_tuner = experiment.runs.get_tuner(run_id='02bab973-ae83-4283-9d73-87b9fd462d35')

```





list()[[source]](_modules/ibm_watsonx_ai/experiment/fm_tune/tune_runs.html#TuneRuns.list)[¶](#ibm_watsonx_ai.experiment.fm_tune.TuneRuns.list "Link to this definition")
Lists historical runs with status. If user has a lot of runs stored in the service,
it may take long time to fetch all the information. If there is no limit set,
get last 50 records.



Returns:
Pandas DataFrame with runs IDs and state



Return type:
pandas.DataFrame




**Examples**



```
from ibm_watsonx_ai.experiment import TuneExperiment

experiment = TuneExperiment(...)
df = experiment.runs.list()

```






Prompt Tuner[¶](#prompt-tuner "Link to this heading")
-----------------------------------------------------




*class* ibm\_watsonx\_ai.foundation\_models.PromptTuner(*name*, *task\_id*, *\**, *description=None*, *base\_model=None*, *accumulate\_steps=None*, *batch\_size=None*, *init\_method=None*, *init\_text=None*, *learning\_rate=None*, *max\_input\_tokens=None*, *max\_output\_tokens=None*, *num\_epochs=None*, *verbalizer=None*, *tuning\_type=None*, *auto\_update\_model=True*, *group\_by\_name=None*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompt_tuner.html#PromptTuner)[¶](#ibm_watsonx_ai.foundation_models.PromptTuner "Link to this definition")
Bases: `object`




cancel\_run(*hard\_delete=False*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompt_tuner.html#PromptTuner.cancel_run)[¶](#ibm_watsonx_ai.foundation_models.PromptTuner.cancel_run "Link to this definition")
Cancels or deletes a Prompt Tuning run.



Parameters:
**hard\_delete** (*bool**,* *optional*) – When True then the completed or cancelled prompt tuning run is deleted,
if False then the current run is canceled. Default: False







get\_data\_connections()[[source]](_modules/ibm_watsonx_ai/foundation_models/prompt_tuner.html#PromptTuner.get_data_connections)[¶](#ibm_watsonx_ai.foundation_models.PromptTuner.get_data_connections "Link to this definition")

Create DataConnection objects for further user usage(eg. to handle data storage connection).





Returns:
list of DataConnections



Return type:
list[‘DataConnection’]




**Example**



```
from ibm_watsonx_ai.experiment import TuneExperiment
experiment = TuneExperiment(credentials, ...)
prompt_tuner = experiment.prompt_tuner(...)
prompt_tuner.run(...)

data_connections = prompt_tuner.get_data_connections()

```





get\_model\_id()[[source]](_modules/ibm_watsonx_ai/foundation_models/prompt_tuner.html#PromptTuner.get_model_id)[¶](#ibm_watsonx_ai.foundation_models.PromptTuner.get_model_id "Link to this definition")
Get model id.


**Example**



```
from ibm_watsonx_ai.experiment import TuneExperiment

experiment = TuneExperiment(credentials, ...)
prompt_tuner = experiment.prompt_tuner(...)
prompt_tuner.run(...)

prompt_tuner.get_model_id()

```





get\_params()[[source]](_modules/ibm_watsonx_ai/foundation_models/prompt_tuner.html#PromptTuner.get_params)[¶](#ibm_watsonx_ai.foundation_models.PromptTuner.get_params "Link to this definition")
Get configuration parameters of PromptTuner.



Returns:
PromptTuner parameters



Return type:
dict




**Example**



```
from ibm_watsonx_ai.experiment import TuneExperiment

experiment = TuneExperiment(credentials, ...)
prompt_tuner = experiment.prompt_tuner(...)

prompt_tuner.get_params()

# Result:
#
# {'base_model': {'name': 'google/flan-t5-xl'},
#  'task_id': 'summarization',
#  'name': 'Prompt Tuning of Flan T5 model',
#  'auto_update_model': False,
#  'group_by_name': False}

```





get\_run\_details(*include\_metrics=False*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompt_tuner.html#PromptTuner.get_run_details)[¶](#ibm_watsonx_ai.foundation_models.PromptTuner.get_run_details "Link to this definition")
Get prompt tuning run details.



Parameters:
**include\_metrics** (*bool**,* *optional*) – indicates to include metrics in the training details output



Returns:
Prompt tuning details



Return type:
dict




**Example**



```
from ibm_watsonx_ai.experiment import TuneExperiment

experiment = TuneExperiment(credentials, ...)
prompt_tuner = experiment.prompt_tuner(...)
prompt_tuner.run(...)

prompt_tuner.get_run_details()

```





get\_run\_status()[[source]](_modules/ibm_watsonx_ai/foundation_models/prompt_tuner.html#PromptTuner.get_run_status)[¶](#ibm_watsonx_ai.foundation_models.PromptTuner.get_run_status "Link to this definition")
Check status/state of initialized Prompt Tuning run if ran in background mode.



Returns:
Prompt tuning run status



Return type:
str




**Example**



```
from ibm_watsonx_ai.experiment import TuneExperiment

experiment = TuneExperiment(credentials, ...)
prompt_tuner = experiment.prompt_tuner(...)
prompt_tuner.run(...)

prompt_tuner.get_run_details()

# Result:
# 'completed'

```





plot\_learning\_curve()[[source]](_modules/ibm_watsonx_ai/foundation_models/prompt_tuner.html#PromptTuner.plot_learning_curve)[¶](#ibm_watsonx_ai.foundation_models.PromptTuner.plot_learning_curve "Link to this definition")
Plot learning curves.



Note


Available only for Jupyter notebooks.



**Example**



```
from ibm_watsonx_ai.experiment import TuneExperiment

experiment = TuneExperiment(credentials, ...)
prompt_tuner = experiment.prompt_tuner(...)
prompt_tuner.run(...)

prompt_tuner.plot_learning_curve()

```





run(*training\_data\_references*, *training\_results\_reference=None*, *background\_mode=False*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompt_tuner.html#PromptTuner.run)[¶](#ibm_watsonx_ai.foundation_models.PromptTuner.run "Link to this definition")
Run a prompt tuning process of foundation model on top of the training data referenced by DataConnection.



Parameters:
* **training\_data\_references** (*list**[*[*DataConnection*](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DataConnection "ibm_watsonx_ai.helpers.connections.connections.DataConnection")*]*) – data storage connection details to inform where training data is stored
* **training\_results\_reference** ([*DataConnection*](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DataConnection "ibm_watsonx_ai.helpers.connections.connections.DataConnection")*,* *optional*) – data storage connection details to store pipeline training results
* **background\_mode** (*bool**,* *optional*) – indicator if fit() method will run in background (async) or (sync)



Returns:
run details



Return type:
dict




**Example**



```
from ibm_watsonx_ai.experiment import TuneExperiment
from ibm_watsonx_ai.helpers import DataConnection, S3Location

experiment = TuneExperiment(credentials, ...)
prompt_tuner = experiment.prompt_tuner(...)

prompt_tuner.run(
    training_data_references=[DataConnection(
        connection_asset_id=connection_id,
        location=S3Location(
            bucket='prompt_tuning_data',
            path='pt_train_data.json')
        )
    )]
    background_mode=False)

```





summary(*scoring='loss'*)[[source]](_modules/ibm_watsonx_ai/foundation_models/prompt_tuner.html#PromptTuner.summary)[¶](#ibm_watsonx_ai.foundation_models.PromptTuner.summary "Link to this definition")
Print PromptTuner models details (prompt-tuned models).



Parameters:
**scoring** (*string**,* *optional*) – scoring metric which user wants to use to sort pipelines by,
when not provided use loss one



Returns:
computed models and metrics



Return type:
pandas.DataFrame




**Example**



```
from ibm_watsonx_ai.experiment import TuneExperiment

experiment = TuneExperiment(credentials, ...)
prompt_tuner = experiment.prompt_tuner(...)
prompt_tuner.run(...)

prompt_tuner.summary()

# Result:
#                          Enhancements            Base model  ...         loss
#       Model Name
# Prompt_tuned_M_1      [prompt_tuning]     google/flan-t5-xl  ...     0.449197

```






Enums[¶](#enums "Link to this heading")
---------------------------------------




*class* ibm\_watsonx\_ai.foundation\_models.utils.enums.PromptTuningTypes[[source]](_modules/ibm_watsonx_ai/foundation_models/utils/enums.html#PromptTuningTypes)[¶](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningTypes "Link to this definition")
Bases: `object`




PT *= 'prompt\_tuning'*[¶](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningTypes.PT "Link to this definition")




*class* ibm\_watsonx\_ai.foundation\_models.utils.enums.PromptTuningInitMethods[[source]](_modules/ibm_watsonx_ai/foundation_models/utils/enums.html#PromptTuningInitMethods)[¶](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningInitMethods "Link to this definition")
Bases: `object`


Supported methods for prompt initialization in prompt tuning.




RANDOM *= 'random'*[¶](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningInitMethods.RANDOM "Link to this definition")



TEXT *= 'text'*[¶](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningInitMethods.TEXT "Link to this definition")




*class* ibm\_watsonx\_ai.foundation\_models.utils.enums.TuneExperimentTasks(*value*)[[source]](_modules/ibm_watsonx_ai/foundation_models/utils/enums.html#TuneExperimentTasks)[¶](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks "Link to this definition")
Bases: `Enum`


An enumeration.




CLASSIFICATION *= 'classification'*[¶](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.CLASSIFICATION "Link to this definition")



CODE\_GENERATION\_AND\_CONVERSION *= 'code'*[¶](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.CODE_GENERATION_AND_CONVERSION "Link to this definition")



EXTRACTION *= 'extraction'*[¶](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.EXTRACTION "Link to this definition")



GENERATION *= 'generation'*[¶](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.GENERATION "Link to this definition")



QUESTION\_ANSWERING *= 'question\_answering'*[¶](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.QUESTION_ANSWERING "Link to this definition")



RETRIEVAL\_AUGMENTED\_GENERATION *= 'retrieval\_augmented\_generation'*[¶](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.RETRIEVAL_AUGMENTED_GENERATION "Link to this definition")



SUMMARIZATION *= 'summarization'*[¶](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.SUMMARIZATION "Link to this definition")




*class* PromptTunableModels[¶](#PromptTunableModels "Link to this definition")
Bases: `Enum`


This represents a dynamically generated Enum for Prompt Tunable Models.


**Example of getting PromptTunableModels**



```
# GET PromptTunableModels ENUM
client.foundation_models.PromptTunableModels

# PRINT dict of Enums
client.foundation_models.PromptTunableModels.show()

```


**Example Output:**



```
{'FLAN_T5_XL': 'google/flan-t5-xl',
'GRANITE_13B_INSTRUCT_V2': 'ibm/granite-13b-instruct-v2',
'LLAMA_2_13B_CHAT': 'meta-llama/llama-2-13b-chat'}

```









[Next

Prompt Template Manager](prompt_template_manager.html)
[Previous

Tuned Model Inference](pt_model_inference.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Tune Experiment](#)
	+ [TuneExperiment](#tuneexperiment)
		- [`TuneExperiment`](#ibm_watsonx_ai.experiment.fm_tune.TuneExperiment)
			* [`TuneExperiment.prompt_tuner()`](#ibm_watsonx_ai.experiment.fm_tune.TuneExperiment.prompt_tuner)
			* [`TuneExperiment.runs()`](#ibm_watsonx_ai.experiment.fm_tune.TuneExperiment.runs)
	+ [Tune Runs](#tune-runs)
		- [`TuneRuns`](#ibm_watsonx_ai.experiment.fm_tune.TuneRuns)
			* [`TuneRuns.get_run_details()`](#ibm_watsonx_ai.experiment.fm_tune.TuneRuns.get_run_details)
			* [`TuneRuns.get_tuner()`](#ibm_watsonx_ai.experiment.fm_tune.TuneRuns.get_tuner)
			* [`TuneRuns.list()`](#ibm_watsonx_ai.experiment.fm_tune.TuneRuns.list)
	+ [Prompt Tuner](#prompt-tuner)
		- [`PromptTuner`](#ibm_watsonx_ai.foundation_models.PromptTuner)
			* [`PromptTuner.cancel_run()`](#ibm_watsonx_ai.foundation_models.PromptTuner.cancel_run)
			* [`PromptTuner.get_data_connections()`](#ibm_watsonx_ai.foundation_models.PromptTuner.get_data_connections)
			* [`PromptTuner.get_model_id()`](#ibm_watsonx_ai.foundation_models.PromptTuner.get_model_id)
			* [`PromptTuner.get_params()`](#ibm_watsonx_ai.foundation_models.PromptTuner.get_params)
			* [`PromptTuner.get_run_details()`](#ibm_watsonx_ai.foundation_models.PromptTuner.get_run_details)
			* [`PromptTuner.get_run_status()`](#ibm_watsonx_ai.foundation_models.PromptTuner.get_run_status)
			* [`PromptTuner.plot_learning_curve()`](#ibm_watsonx_ai.foundation_models.PromptTuner.plot_learning_curve)
			* [`PromptTuner.run()`](#ibm_watsonx_ai.foundation_models.PromptTuner.run)
			* [`PromptTuner.summary()`](#ibm_watsonx_ai.foundation_models.PromptTuner.summary)
	+ [Enums](#enums)
		- [`PromptTuningTypes`](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningTypes)
			* [`PromptTuningTypes.PT`](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningTypes.PT)
		- [`PromptTuningInitMethods`](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningInitMethods)
			* [`PromptTuningInitMethods.RANDOM`](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningInitMethods.RANDOM)
			* [`PromptTuningInitMethods.TEXT`](#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningInitMethods.TEXT)
		- [`TuneExperimentTasks`](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks)
			* [`TuneExperimentTasks.CLASSIFICATION`](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.CLASSIFICATION)
			* [`TuneExperimentTasks.CODE_GENERATION_AND_CONVERSION`](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.CODE_GENERATION_AND_CONVERSION)
			* [`TuneExperimentTasks.EXTRACTION`](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.EXTRACTION)
			* [`TuneExperimentTasks.GENERATION`](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.GENERATION)
			* [`TuneExperimentTasks.QUESTION_ANSWERING`](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.QUESTION_ANSWERING)
			* [`TuneExperimentTasks.RETRIEVAL_AUGMENTED_GENERATION`](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.RETRIEVAL_AUGMENTED_GENERATION)
			* [`TuneExperimentTasks.SUMMARIZATION`](#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.SUMMARIZATION)
		- [`PromptTunableModels`](#PromptTunableModels)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/autoai_deployment_modules.html








Deployment Modules for AutoAI models - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](#)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Deployment Modules for AutoAI models[¶](#deployment-modules-for-autoai-models "Link to this heading")
=====================================================================================================



Web Service[¶](#web-service "Link to this heading")
---------------------------------------------------


For usage instruction see [Web Service](autoai_working_with_class_and_optimizer.html#working-with-web-service).




*class* ibm\_watsonx\_ai.deployment.WebService(*source\_instance\_credentials=None*, *source\_project\_id=None*, *source\_space\_id=None*, *target\_instance\_credentials=None*, *target\_project\_id=None*, *target\_space\_id=None*, *project\_id=None*, *space\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/deployment/web_service.html#WebService)[¶](#ibm_watsonx_ai.deployment.WebService "Link to this definition")
Bases: `BaseDeployment`


An Online Deployment class aka. WebService.
With this class object you can manage any online (WebService) deployment.



Parameters:
* **source\_instance\_credentials** (*dict*) – credentials to the instance where training was performed
* **source\_project\_id** (*str**,* *optional*) – ID of the Watson Studio project where training was performed
* **source\_space\_id** (*str**,* *optional*) – ID of the Watson Studio Space where training was performed
* **target\_instance\_credentials** (*dict*) – credentials to the instance where you want to deploy
* **target\_project\_id** (*str**,* *optional*) – ID of the Watson Studio project where you want to deploy
* **target\_space\_id** (*str**,* *optional*) – ID of the Watson Studio Space where you want to deploy






create(*model*, *deployment\_name*, *serving\_name=None*, *metadata=None*, *training\_data=None*, *training\_target=None*, *experiment\_run\_id=None*, *hardware\_spec=None*)[[source]](_modules/ibm_watsonx_ai/deployment/web_service.html#WebService.create)[¶](#ibm_watsonx_ai.deployment.WebService.create "Link to this definition")
Create deployment from a model.



Parameters:
* **model** (*str*) – AutoAI model name
* **deployment\_name** (*str*) – name of the deployment
* **training\_data** (*pandas.DataFrame* *or* *numpy.ndarray**,* *optional*) – training data for the model
* **training\_target** (*pandas.DataFrame* *or* *numpy.ndarray**,* *optional*) – target/label data for the model
* **serving\_name** (*str**,* *optional*) – serving name of the deployment
* **metadata** (*dict**,* *optional*) – model meta properties
* **experiment\_run\_id** (*str**,* *optional*) – ID of a training/experiment (only applicable for AutoAI deployments)
* **hardware\_spec** (*dict**,* *optional*) – hardware specification for deployment




**Example**



```
from ibm_watsonx_ai.deployment import WebService
from ibm_watsonx_ai import Credentials

deployment = WebService(
        source_instance_credentials=Credentials(...),
        source_project_id="...",
        target_space_id="...")

deployment.create(
       experiment_run_id="...",
       model=model,
       deployment_name='My new deployment',
       serving_name='my_new_deployment'
   )

```





delete(*deployment\_id=None*)[[source]](_modules/ibm_watsonx_ai/deployment/web_service.html#WebService.delete)[¶](#ibm_watsonx_ai.deployment.WebService.delete "Link to this definition")
Delete deployment.



Parameters:
**deployment\_id** (*str**,* *optional*) – ID of the deployment to delete, if empty, current deployment will be deleted




**Example**



```
deployment = WebService(workspace=...)
# Delete current deployment
deployment.delete()
# Or delete a specific deployment
deployment.delete(deployment_id='...')

```





get(*deployment\_id*)[[source]](_modules/ibm_watsonx_ai/deployment/web_service.html#WebService.get)[¶](#ibm_watsonx_ai.deployment.WebService.get "Link to this definition")
Get deployment.



Parameters:
**deployment\_id** (*str*) – ID of the deployment to work with




**Example**



```
deployment = WebService(workspace=...)
deployment.get(deployment_id="...")

```





get\_params()[[source]](_modules/ibm_watsonx_ai/deployment/web_service.html#WebService.get_params)[¶](#ibm_watsonx_ai.deployment.WebService.get_params "Link to this definition")
Get deployment parameters.





list(*limit=None*)[[source]](_modules/ibm_watsonx_ai/deployment/web_service.html#WebService.list)[¶](#ibm_watsonx_ai.deployment.WebService.list "Link to this definition")
List deployments.



Parameters:
**limit** (*int**,* *optional*) – set the limit of how many deployments to list,
default is None (all deployments should be fetched)



Returns:
Pandas DataFrame with information about deployments



Return type:
pandas.DataFrame




**Example**



```
deployment = WebService(workspace=...)
deployments_list = deployment.list()
print(deployments_list)

# Result:
#                  created_at  ...  status
# 0  2020-03-06T10:50:49.401Z  ...   ready
# 1  2020-03-06T13:16:09.789Z  ...   ready
# 4  2020-03-11T14:46:36.035Z  ...  failed
# 3  2020-03-11T14:49:55.052Z  ...  failed
# 2  2020-03-11T15:13:53.708Z  ...   ready

```





score(*payload=Empty DataFrame Columns: [] Index: []*, *\**, *forecast\_window=None*, *transaction\_id=None*)[[source]](_modules/ibm_watsonx_ai/deployment/web_service.html#WebService.score)[¶](#ibm_watsonx_ai.deployment.WebService.score "Link to this definition")
Online scoring. Payload is passed to the Service scoring endpoint where model have been deployed.



Parameters:
* **payload** (*pandas.DataFrame* *or* *dict*) – DataFrame with data to test the model or dictionary with keys observations
and supporting\_features and DataFrames with data for observations and supporting\_features
to score forecasting models
* **forecast\_window** (*int**,* *optional*) – size of forecast window, supported only for forcasting, supported from CPD 5.0
* **transaction\_id** (*str**,* *optional*) – can be used to indicate under which id the records will be saved into payload table
in IBM OpenScale



Returns:
dictionary with list od model output/predicted targets



Return type:
dict




**Examples**



```
predictions = web_service.score(payload=test_data)
print(predictions)

# Result:
# {'predictions':
#     [{
#         'fields': ['prediction', 'probability'],
#         'values': [['no', [0.9221385608558003, 0.07786143914419975]],
#                   ['no', [0.9798324002736079, 0.020167599726392187]]
#     }]}

predictions = web_service.score(payload={'observations': new_observations_df})
predictions = web_service.score(payload={'observations': new_observations_df, 'supporting_features': supporting_features_df}) # supporting features time series forecasting scenario
predictions = web_service.score(payload={'observations': new_observations_df}
                                forecast_window=1000) # forecast_window time series forecasting scenario

```






Batch[¶](#batch "Link to this heading")
---------------------------------------


For usage instruction see [Batch](autoai_working_with_class_and_optimizer.html#working-with-batch).




*class* ibm\_watsonx\_ai.deployment.Batch(*source\_instance\_credentials=None*, *source\_project\_id=None*, *source\_space\_id=None*, *target\_instance\_credentials=None*, *target\_project\_id=None*, *target\_space\_id=None*, *project\_id=None*, *space\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/deployment/batch.html#Batch)[¶](#ibm_watsonx_ai.deployment.Batch "Link to this definition")
Bases: `BaseDeployment`


The Batch Deployment class.
With this class object you can manage any batch deployment.



Parameters:
* **source\_instance\_credentials** (*dict*) – credentials to the instance where training was performed
* **source\_project\_id** (*str**,* *optional*) – ID of the Watson Studio project where training was performed
* **source\_space\_id** (*str**,* *optional*) – ID of the Watson Studio Space where training was performed
* **target\_instance\_credentials** (*dict*) – credentials to the instance where you want to deploy
* **target\_project\_id** (*str**,* *optional*) – ID of the Watson Studio project where you want to deploy
* **target\_space\_id** (*str**,* *optional*) – ID of the Watson Studio Space where you want to deploy






create(*model*, *deployment\_name*, *metadata=None*, *training\_data=None*, *training\_target=None*, *experiment\_run\_id=None*, *hardware\_spec=None*)[[source]](_modules/ibm_watsonx_ai/deployment/batch.html#Batch.create)[¶](#ibm_watsonx_ai.deployment.Batch.create "Link to this definition")
Create deployment from a model.



Parameters:
* **model** (*str*) – AutoAI model name
* **deployment\_name** (*str*) – name of the deployment
* **training\_data** (*pandas.DataFrame* *or* *numpy.ndarray**,* *optional*) – training data for the model
* **training\_target** (*pandas.DataFrame* *or* *numpy.ndarray**,* *optional*) – target/label data for the model
* **metadata** (*dict**,* *optional*) – model meta properties
* **experiment\_run\_id** (*str**,* *optional*) – ID of a training/experiment (only applicable for AutoAI deployments)
* **hardware\_spec** (*str**,* *optional*) – hardware specification name of the deployment




**Example**



```
from ibm_watsonx_ai.deployment import Batch

deployment = Batch(
        source_instance_credentials=Credentials(...),
        source_project_id="...",
        target_space_id="...")

deployment.create(
       experiment_run_id="...",
       model=model,
       deployment_name='My new deployment'
       hardware_spec='L'
   )

```





delete(*deployment\_id=None*)[[source]](_modules/ibm_watsonx_ai/deployment/batch.html#Batch.delete)[¶](#ibm_watsonx_ai.deployment.Batch.delete "Link to this definition")
Delete deployment.



Parameters:
**deployment\_id** (*str**,* *optional*) – ID of the deployment to delete, if empty, current deployment will be deleted




**Example**



```
deployment = Batch(workspace=...)
# Delete current deployment
deployment.delete()
# Or delete a specific deployment
deployment.delete(deployment_id='...')

```





get(*deployment\_id*)[[source]](_modules/ibm_watsonx_ai/deployment/batch.html#Batch.get)[¶](#ibm_watsonx_ai.deployment.Batch.get "Link to this definition")
Get deployment.



Parameters:
**deployment\_id** (*str*) – ID of the deployment to work with




**Example**



```
deployment = Batch(workspace=...)
deployment.get(deployment_id="...")

```





get\_job\_id(*batch\_scoring\_details*)[[source]](_modules/ibm_watsonx_ai/deployment/batch.html#Batch.get_job_id)[¶](#ibm_watsonx_ai.deployment.Batch.get_job_id "Link to this definition")
Get id from batch scoring details.





get\_job\_params(*scoring\_job\_id=None*)[[source]](_modules/ibm_watsonx_ai/deployment/batch.html#Batch.get_job_params)[¶](#ibm_watsonx_ai.deployment.Batch.get_job_params "Link to this definition")
Get batch deployment job parameters.



Parameters:
**scoring\_job\_id** (*str*) – Id of scoring job



Returns:
parameters of the scoring job



Return type:
dict







get\_job\_result(*scoring\_job\_id*)[[source]](_modules/ibm_watsonx_ai/deployment/batch.html#Batch.get_job_result)[¶](#ibm_watsonx_ai.deployment.Batch.get_job_result "Link to this definition")
Get batch deployment results of job with id scoring\_job\_id.



Parameters:
**scoring\_job\_id** (*str*) – Id of scoring job which results will be returned



Returns:
result



Return type:
pandas.DataFrame



Raises:
**MissingScoringResults** – in case of incompleted or failed job
MissingScoringResults scoring exception is raised







get\_job\_status(*scoring\_job\_id*)[[source]](_modules/ibm_watsonx_ai/deployment/batch.html#Batch.get_job_status)[¶](#ibm_watsonx_ai.deployment.Batch.get_job_status "Link to this definition")
Get status of scoring job.



Parameters:
**scoring\_job\_id** (*str*) – Id of scoring job



Returns:
dictionary with state of scoring job (one of: [completed, failed, starting, queued])
and additional details if they exist



Return type:
dict







get\_params()[[source]](_modules/ibm_watsonx_ai/deployment/batch.html#Batch.get_params)[¶](#ibm_watsonx_ai.deployment.Batch.get_params "Link to this definition")
Get deployment parameters.





list(*limit=None*)[[source]](_modules/ibm_watsonx_ai/deployment/batch.html#Batch.list)[¶](#ibm_watsonx_ai.deployment.Batch.list "Link to this definition")
List deployments.



Parameters:
**limit** (*int**,* *optional*) – set the limit of how many deployments to list,
default is None (all deployments should be fetched)



Returns:
Pandas DataFrame with information about deployments



Return type:
pandas.DataFrame




**Example**



```
deployment = Batch(workspace=...)
deployments_list = deployment.list()
print(deployments_list)

# Result:
#                  created_at  ...  status
# 0  2020-03-06T10:50:49.401Z  ...   ready
# 1  2020-03-06T13:16:09.789Z  ...   ready
# 4  2020-03-11T14:46:36.035Z  ...  failed
# 3  2020-03-11T14:49:55.052Z  ...  failed
# 2  2020-03-11T15:13:53.708Z  ...   ready

```





list\_jobs()[[source]](_modules/ibm_watsonx_ai/deployment/batch.html#Batch.list_jobs)[¶](#ibm_watsonx_ai.deployment.Batch.list_jobs "Link to this definition")
Returns pandas DataFrame with list of deployment jobs





rerun\_job(*scoring\_job\_id*, *background\_mode=True*)[[source]](_modules/ibm_watsonx_ai/deployment/batch.html#Batch.rerun_job)[¶](#ibm_watsonx_ai.deployment.Batch.rerun_job "Link to this definition")
Rerun scoring job with the same parameters as job described by scoring\_job\_id.



Parameters:
* **scoring\_job\_id** (*str*) – Id described scoring job
* **background\_mode** (*bool**,* *optional*) – indicator if score\_rerun() method will run in background (async) or (sync)



Returns:
scoring job details



Return type:
dict




**Example**



```
scoring_details = deployment.score_rerun(scoring_job_id)

```





run\_job(*payload=Empty DataFrame Columns: [] Index: []*, *output\_data\_reference=None*, *transaction\_id=None*, *background\_mode=True*, *hardware\_spec=None*)[[source]](_modules/ibm_watsonx_ai/deployment/batch.html#Batch.run_job)[¶](#ibm_watsonx_ai.deployment.Batch.run_job "Link to this definition")
Batch scoring job. Payload or Payload data reference is required.
It is passed to the Service where model have been deployed.



Parameters:
* **payload** (*pandas.DataFrame* *or* *List**[*[*DataConnection*](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DataConnection "ibm_watsonx_ai.helpers.connections.connections.DataConnection")*] or* *Dict*) – DataFrame that contains data to test the model or data storage connection details
that inform the model where payload data is stored
* **output\_data\_reference** ([*DataConnection*](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DataConnection "ibm_watsonx_ai.helpers.connections.connections.DataConnection")*,* *optional*) – DataConnection to the output COS for storing predictions,
required only when DataConnections are used as a payload
* **transaction\_id** (*str**,* *optional*) – can be used to indicate under which id the records will be saved into payload table
in IBM OpenScale
* **background\_mode** (*bool**,* *optional*) – indicator if score() method will run in background (async) or (sync)
* **hardware\_spec** (*str**,* *optional*) – hardware specification name for scoring job



Returns:
scoring job details



Return type:
dict




**Examples**



```
score_details = batch_service.run_job(payload=test_data)
print(score_details['entity']['scoring'])

# Result:
# {'input_data': [{'fields': ['sepal_length',
#               'sepal_width',
#               'petal_length',
#               'petal_width'],
#              'values': [[4.9, 3.0, 1.4, 0.2]]}],
# 'predictions': [{'fields': ['prediction', 'probability'],
#               'values': [['setosa',
#                 [0.9999320742502246,
#                  5.1519823540224506e-05,
#                  1.6405926235405522e-05]]]}]

payload_reference = DataConnection(location=DSLocation(asset_id=asset_id))
score_details = batch_service.run_job(payload=payload_reference, output_data_filename = "scoring_output.csv")
score_details = batch_service.run_job(payload={'observations': payload_reference})
score_details = batch_service.run_job(payload=[payload_reference])
score_details = batch_service.run_job(payload={'observations': payload_reference, 'supporting_features': supporting_features_reference})  # supporting features time series forecasting sceanrio
score_details = batch_service.run_job(payload=test_df, hardware_spec='S')
score_details = batch_service.run_job(payload=test_df, hardware_spec=TShirtSize.L)

```





score(*\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/deployment/batch.html#Batch.score)[¶](#ibm_watsonx_ai.deployment.Batch.score "Link to this definition")
Scoring on Service. Payload is passed to the scoring endpoint where model have been deployed.



Parameters:
**payload** (*pandas.DataFrame*) – data to test the model












[Next

Foundation Models](foundation_models.html)
[Previous

AutoAI experiment](autoai_experiment.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Deployment Modules for AutoAI models](#)
	+ [Web Service](#web-service)
		- [`WebService`](#ibm_watsonx_ai.deployment.WebService)
			* [`WebService.create()`](#ibm_watsonx_ai.deployment.WebService.create)
			* [`WebService.delete()`](#ibm_watsonx_ai.deployment.WebService.delete)
			* [`WebService.get()`](#ibm_watsonx_ai.deployment.WebService.get)
			* [`WebService.get_params()`](#ibm_watsonx_ai.deployment.WebService.get_params)
			* [`WebService.list()`](#ibm_watsonx_ai.deployment.WebService.list)
			* [`WebService.score()`](#ibm_watsonx_ai.deployment.WebService.score)
	+ [Batch](#batch)
		- [`Batch`](#ibm_watsonx_ai.deployment.Batch)
			* [`Batch.create()`](#ibm_watsonx_ai.deployment.Batch.create)
			* [`Batch.delete()`](#ibm_watsonx_ai.deployment.Batch.delete)
			* [`Batch.get()`](#ibm_watsonx_ai.deployment.Batch.get)
			* [`Batch.get_job_id()`](#ibm_watsonx_ai.deployment.Batch.get_job_id)
			* [`Batch.get_job_params()`](#ibm_watsonx_ai.deployment.Batch.get_job_params)
			* [`Batch.get_job_result()`](#ibm_watsonx_ai.deployment.Batch.get_job_result)
			* [`Batch.get_job_status()`](#ibm_watsonx_ai.deployment.Batch.get_job_status)
			* [`Batch.get_params()`](#ibm_watsonx_ai.deployment.Batch.get_params)
			* [`Batch.list()`](#ibm_watsonx_ai.deployment.Batch.list)
			* [`Batch.list_jobs()`](#ibm_watsonx_ai.deployment.Batch.list_jobs)
			* [`Batch.rerun_job()`](#ibm_watsonx_ai.deployment.Batch.rerun_job)
			* [`Batch.run_job()`](#ibm_watsonx_ai.deployment.Batch.run_job)
			* [`Batch.score()`](#ibm_watsonx_ai.deployment.Batch.score)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/foundation_models.html








Foundation Models - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](#)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Foundation Models[¶](#foundation-models "Link to this heading")
===============================================================



Warning


Warning! Supported only for IBM watsonx.ai for IBM Cloud and IBM watsonx.ai software with IBM Cloud Pak for Data 4.8 and higher.




Modules[¶](#modules "Link to this heading")
-------------------------------------------



* [Embeddings](fm_embeddings.html)
	+ [Embeddings](fm_embeddings.html#id1)
		- [`Embeddings`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.Embeddings)
			* [`Embeddings.embed_documents()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.embed_documents)
			* [`Embeddings.embed_query()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.embed_query)
			* [`Embeddings.generate()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.generate)
			* [`Embeddings.to_dict()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.to_dict)
	+ [BaseEmbeddings](fm_embeddings.html#baseembeddings)
		- [`BaseEmbeddings`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings)
			* [`BaseEmbeddings.embed_documents()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.embed_documents)
			* [`BaseEmbeddings.embed_query()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.embed_query)
			* [`BaseEmbeddings.from_dict()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.from_dict)
			* [`BaseEmbeddings.to_dict()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.to_dict)
	+ [Enums](fm_embeddings.html#enums)
		- [`EmbedTextParamsMetaNames`](fm_embeddings.html#metanames.EmbedTextParamsMetaNames)
		- [`EmbeddingModels`](fm_embeddings.html#EmbeddingModels)
		- [`EmbeddingTypes`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.utils.enums.EmbeddingTypes)
* [Models](fm_models.html)
	+ [Modules](fm_models.html#modules)
		- [Model](fm_model.html)
			* [`Model`](fm_model.html#ibm_watsonx_ai.foundation_models.Model)
				+ [`Model.generate()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.generate)
				+ [`Model.generate_text()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.generate_text)
				+ [`Model.generate_text_stream()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.generate_text_stream)
				+ [`Model.get_details()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.get_details)
				+ [`Model.to_langchain()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.to_langchain)
				+ [`Model.tokenize()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.tokenize)
			* [Enums](fm_model.html#enums)
				+ [`GenTextParamsMetaNames`](fm_model.html#metanames.GenTextParamsMetaNames)
				+ [`GenTextReturnOptMetaNames`](fm_model.html#metanames.GenTextReturnOptMetaNames)
				+ [`DecodingMethods`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods)
					- [`DecodingMethods.GREEDY`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods.GREEDY)
					- [`DecodingMethods.SAMPLE`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods.SAMPLE)
				+ [`TextModels`](fm_model.html#TextModels)
				+ [`ModelTypes`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes)
		- [ModelInference](fm_model_inference.html)
			* [`ModelInference`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference)
				+ [`ModelInference.generate()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate)
				+ [`ModelInference.generate_text()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate_text)
				+ [`ModelInference.generate_text_stream()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate_text_stream)
				+ [`ModelInference.get_details()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.get_details)
				+ [`ModelInference.get_identifying_params()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.get_identifying_params)
				+ [`ModelInference.to_langchain()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.to_langchain)
				+ [`ModelInference.tokenize()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.tokenize)
		- [`ModelInference` for Deployments](fm_deployments.html)
			* [Infer text with deployments](fm_deployments.html#infer-text-with-deployments)
			* [Creating `ModelInference` instance](fm_deployments.html#creating-modelinference-instance)
			* [Generate methods](fm_deployments.html#generate-methods)
* [Prompt Tuning](prompt_tuner.html)
	+ [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)
		- [Tune Experiment run](pt_tune_experiment_run.html)
			* [Configure PromptTuner](pt_tune_experiment_run.html#configure-prompttuner)
			* [Get configuration parameters](pt_tune_experiment_run.html#get-configuration-parameters)
			* [Run prompt tuning](pt_tune_experiment_run.html#run-prompt-tuning)
			* [Get run status, get run details](pt_tune_experiment_run.html#get-run-status-get-run-details)
			* [Get data connections](pt_tune_experiment_run.html#get-data-connections)
			* [Summary](pt_tune_experiment_run.html#summary)
			* [Plot learning curves](pt_tune_experiment_run.html#plot-learning-curves)
			* [Get model identifier](pt_tune_experiment_run.html#get-model-identifier)
		- [Tuned Model Inference](pt_model_inference.html)
			* [Working with deployments](pt_model_inference.html#working-with-deployments)
			* [Creating `ModelInference` instance](pt_model_inference.html#creating-modelinference-instance)
			* [Importing data](pt_model_inference.html#importing-data)
			* [Analyzing satisfaction](pt_model_inference.html#analyzing-satisfaction)
			* [Generate methods](pt_model_inference.html#generate-methods)
	+ [Tune Experiment](tune_experiment.html)
		- [TuneExperiment](tune_experiment.html#tuneexperiment)
			* [`TuneExperiment`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneExperiment)
				+ [`TuneExperiment.prompt_tuner()`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneExperiment.prompt_tuner)
				+ [`TuneExperiment.runs()`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneExperiment.runs)
		- [Tune Runs](tune_experiment.html#tune-runs)
			* [`TuneRuns`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneRuns)
				+ [`TuneRuns.get_run_details()`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneRuns.get_run_details)
				+ [`TuneRuns.get_tuner()`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneRuns.get_tuner)
				+ [`TuneRuns.list()`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneRuns.list)
		- [Prompt Tuner](tune_experiment.html#prompt-tuner)
			* [`PromptTuner`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner)
				+ [`PromptTuner.cancel_run()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.cancel_run)
				+ [`PromptTuner.get_data_connections()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.get_data_connections)
				+ [`PromptTuner.get_model_id()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.get_model_id)
				+ [`PromptTuner.get_params()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.get_params)
				+ [`PromptTuner.get_run_details()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.get_run_details)
				+ [`PromptTuner.get_run_status()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.get_run_status)
				+ [`PromptTuner.plot_learning_curve()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.plot_learning_curve)
				+ [`PromptTuner.run()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.run)
				+ [`PromptTuner.summary()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.summary)
		- [Enums](tune_experiment.html#enums)
			* [`PromptTuningTypes`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningTypes)
				+ [`PromptTuningTypes.PT`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningTypes.PT)
			* [`PromptTuningInitMethods`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningInitMethods)
				+ [`PromptTuningInitMethods.RANDOM`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningInitMethods.RANDOM)
				+ [`PromptTuningInitMethods.TEXT`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningInitMethods.TEXT)
			* [`TuneExperimentTasks`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks)
				+ [`TuneExperimentTasks.CLASSIFICATION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.CLASSIFICATION)
				+ [`TuneExperimentTasks.CODE_GENERATION_AND_CONVERSION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.CODE_GENERATION_AND_CONVERSION)
				+ [`TuneExperimentTasks.EXTRACTION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.EXTRACTION)
				+ [`TuneExperimentTasks.GENERATION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.GENERATION)
				+ [`TuneExperimentTasks.QUESTION_ANSWERING`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.QUESTION_ANSWERING)
				+ [`TuneExperimentTasks.RETRIEVAL_AUGMENTED_GENERATION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.RETRIEVAL_AUGMENTED_GENERATION)
				+ [`TuneExperimentTasks.SUMMARIZATION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.SUMMARIZATION)
			* [`PromptTunableModels`](tune_experiment.html#PromptTunableModels)
* [Prompt Template Manager](prompt_template_manager.html)
	+ [`PromptTemplateManager`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager)
		- [`PromptTemplateManager.delete_prompt()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.delete_prompt)
		- [`PromptTemplateManager.get_lock()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.get_lock)
		- [`PromptTemplateManager.list()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.list)
		- [`PromptTemplateManager.load_prompt()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.load_prompt)
		- [`PromptTemplateManager.lock()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.lock)
		- [`PromptTemplateManager.store_prompt()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.store_prompt)
		- [`PromptTemplateManager.unlock()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.unlock)
		- [`PromptTemplateManager.update_prompt()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.update_prompt)
	+ [`PromptTemplate`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplate)
	+ [`FreeformPromptTemplate`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.FreeformPromptTemplate)
	+ [`DetachedPromptTemplate`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.DetachedPromptTemplate)
	+ [Enums](prompt_template_manager.html#enums)
		- [`PromptTemplateFormats`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats)
			* [`PromptTemplateFormats.LANGCHAIN`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats.LANGCHAIN)
			* [`PromptTemplateFormats.PROMPTTEMPLATE`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats.PROMPTTEMPLATE)
			* [`PromptTemplateFormats.STRING`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats.STRING)
* [Extensions](fm_extensions.html)
	+ [LangChain](fm_extensions.html#langchain)
		- [`WatsonxLLM`](fm_extensions.html#langchain_ibm.WatsonxLLM)
			* [`WatsonxLLM.apikey`](fm_extensions.html#langchain_ibm.WatsonxLLM.apikey)
			* [`WatsonxLLM.deployment_id`](fm_extensions.html#langchain_ibm.WatsonxLLM.deployment_id)
			* [`WatsonxLLM.get_num_tokens()`](fm_extensions.html#langchain_ibm.WatsonxLLM.get_num_tokens)
			* [`WatsonxLLM.get_token_ids()`](fm_extensions.html#langchain_ibm.WatsonxLLM.get_token_ids)
			* [`WatsonxLLM.instance_id`](fm_extensions.html#langchain_ibm.WatsonxLLM.instance_id)
			* [`WatsonxLLM.model_id`](fm_extensions.html#langchain_ibm.WatsonxLLM.model_id)
			* [`WatsonxLLM.params`](fm_extensions.html#langchain_ibm.WatsonxLLM.params)
			* [`WatsonxLLM.password`](fm_extensions.html#langchain_ibm.WatsonxLLM.password)
			* [`WatsonxLLM.project_id`](fm_extensions.html#langchain_ibm.WatsonxLLM.project_id)
			* [`WatsonxLLM.space_id`](fm_extensions.html#langchain_ibm.WatsonxLLM.space_id)
			* [`WatsonxLLM.streaming`](fm_extensions.html#langchain_ibm.WatsonxLLM.streaming)
			* [`WatsonxLLM.token`](fm_extensions.html#langchain_ibm.WatsonxLLM.token)
			* [`WatsonxLLM.url`](fm_extensions.html#langchain_ibm.WatsonxLLM.url)
			* [`WatsonxLLM.username`](fm_extensions.html#langchain_ibm.WatsonxLLM.username)
			* [`WatsonxLLM.verify`](fm_extensions.html#langchain_ibm.WatsonxLLM.verify)
			* [`WatsonxLLM.version`](fm_extensions.html#langchain_ibm.WatsonxLLM.version)
* [Helpers](fm_helpers.html)
	+ [`FoundationModelsManager`](fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager)
		- [`FoundationModelsManager.get_custom_model_specs()`](fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_custom_model_specs)
		- [`FoundationModelsManager.get_embeddings_model_specs()`](fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_embeddings_model_specs)
		- [`FoundationModelsManager.get_model_lifecycle()`](fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_model_lifecycle)
		- [`FoundationModelsManager.get_model_specs()`](fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_model_specs)
		- [`FoundationModelsManager.get_model_specs_with_prompt_tuning_support()`](fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_model_specs_with_prompt_tuning_support)
	+ [`get_model_specs()`](fm_helpers.html#ibm_watsonx_ai.foundation_models.get_model_specs)
	+ [`get_model_lifecycle()`](fm_helpers.html#ibm_watsonx_ai.foundation_models.get_model_lifecycle)
	+ [`get_model_specs_with_prompt_tuning_support()`](fm_helpers.html#ibm_watsonx_ai.foundation_models.get_model_specs_with_prompt_tuning_support)
	+ [`get_supported_tasks()`](fm_helpers.html#ibm_watsonx_ai.foundation_models.get_supported_tasks)
* [Custom models](fm_working_with_custom_models.html)
	+ [Initialize APIClient object](fm_working_with_custom_models.html#initialize-apiclient-object)
	+ [Listing models specification](fm_working_with_custom_models.html#listing-models-specification)
	+ [Storing model in service repository](fm_working_with_custom_models.html#storing-model-in-service-repository)
	+ [Defining hardware specification](fm_working_with_custom_models.html#defining-hardware-specification)
	+ [Deployment of custom foundation model](fm_working_with_custom_models.html#deployment-of-custom-foundation-model)
	+ [Working with deployments](fm_working_with_custom_models.html#working-with-deployments)









[Next

Embeddings](fm_embeddings.html)
[Previous

Deployment Modules for AutoAI models](autoai_deployment_modules.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Foundation Models](#)
	+ [Modules](#modules)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/core_api.html








Core - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](#)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Core[¶](#core "Link to this heading")
=====================================



Connections[¶](#connections "Link to this heading")
---------------------------------------------------




*class* client.Connections(*client*)[[source]](_modules/ibm_watsonx_ai/connections.html#Connections)[¶](#client.Connections "Link to this definition")
Store and manage connections.




ConfigurationMetaNames *= <ibm\_watsonx\_ai.metanames.ConnectionMetaNames object>*[¶](#client.Connections.ConfigurationMetaNames "Link to this definition")
MetaNames for connection creation.





create(*meta\_props*)[[source]](_modules/ibm_watsonx_ai/connections.html#Connections.create)[¶](#client.Connections.create "Link to this definition")
Create a connection. Examples of PROPERTIES field input:


1. MySQL



> ```
> client.connections.ConfigurationMetaNames.PROPERTIES: {
>     "database": "database",
>     "password": "password",
>     "port": "3306",
>     "host": "host url",
>     "ssl": "false",
>     "username": "username"
> }
> 
> ```
2. Google BigQuery



> 1. Method 1: Using service account json. The generated service account json can be provided as input as-is. Provide actual values in json. The example below is only indicative to show the fields. For information on how to generate the service account json, refer to Google BigQuery documentation.
> 	
> 	
> 	
> 	> ```
> 	> client.connections.ConfigurationMetaNames.PROPERTIES: {
> 	>     "type": "service_account",
> 	>     "project_id": "project_id",
> 	>     "private_key_id": "private_key_id",
> 	>     "private_key": "private key contents",
> 	>     "client_email": "client_email",
> 	>     "client_id": "client_id",
> 	>     "auth_uri": "https://accounts.google.com/o/oauth2/auth",
> 	>     "token_uri": "https://oauth2.googleapis.com/token",
> 	>     "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
> 	>     "client_x509_cert_url": "client_x509_cert_url"
> 	> }
> 	> 
> 	> ```
> 	2. Method 2: Using OAuth Method. For information on how to generate a OAuth token, refer to Google BigQuery documentation.
> 	
> 	
> 	
> 	> ```
> 	> client.connections.ConfigurationMetaNames.PROPERTIES: {
> 	>     "access_token": "access token generated for big query",
> 	>     "refresh_token": "refresh token",
> 	>     "project_id": "project_id",
> 	>     "client_secret": "This is your gmail account password",
> 	>     "client_id": "client_id"
> 	> }
> 	> 
> 	> ```
3. MS SQL



> ```
> client.connections.ConfigurationMetaNames.PROPERTIES: {
>     "database": "database",
>     "password": "password",
>     "port": "1433",
>     "host": "host",
>     "username": "username"
> }
> 
> ```
4. Teradata



> ```
> client.connections.ConfigurationMetaNames.PROPERTIES: {
>     "database": "database",
>     "password": "password",
>     "port": "1433",
>     "host": "host",
>     "username": "username"
> }
> 
> ```



Parameters:
**meta\_props** (*dict*) – metadata of the connection configuration. To see available meta names, use:



```
client.connections.ConfigurationMetaNames.get()

```






Returns:
metadata of the stored connection



Return type:
dict




**Example**



```
sqlserver_data_source_type_id = client.connections.get_datasource_type_id_by_name('sqlserver')
connections_details = client.connections.create({
    client.connections.ConfigurationMetaNames.NAME: "sqlserver connection",
    client.connections.ConfigurationMetaNames.DESCRIPTION: "connection description",
    client.connections.ConfigurationMetaNames.DATASOURCE_TYPE: sqlserver_data_source_type_id,
    client.connections.ConfigurationMetaNames.PROPERTIES: { "database": "database",
                                                            "password": "password",
                                                            "port": "1433",
                                                            "host": "host",
                                                            "username": "username"}
})

```





delete(*connection\_id*)[[source]](_modules/ibm_watsonx_ai/connections.html#Connections.delete)[¶](#client.Connections.delete "Link to this definition")
Delete a stored connection.



Parameters:
**connection\_id** (*str*) – unique ID of the connection to be deleted



Returns:
status (“SUCCESS” or “FAILED”)



Return type:
str




**Example**



```
client.connections.delete(connection_id)

```





get\_datasource\_type\_details\_by\_id(*datasource\_type\_id*, *connection\_properties=False*)[[source]](_modules/ibm_watsonx_ai/connections.html#Connections.get_datasource_type_details_by_id)[¶](#client.Connections.get_datasource_type_details_by_id "Link to this definition")
Get datasource type details for the given datasource type ID.



Parameters:
* **datasource\_type\_id** (*str*) – ID of the datasource type
* **connection\_properties** (*bool*) – if True, the connection properties are included in the returned details. defaults to False



Returns:
Datasource type details



Return type:
dict




**Example**



```
client.connections.get_datasource_type_details_by_id(datasource_type_id)

```





get\_datasource\_type\_id\_by\_name(*name*)[[source]](_modules/ibm_watsonx_ai/connections.html#Connections.get_datasource_type_id_by_name)[¶](#client.Connections.get_datasource_type_id_by_name "Link to this definition")
Get a stored datasource type ID for the given datasource type name.



Parameters:
**name** (*str*) – name of datasource type



Returns:
ID of datasource type



Return type:
str




**Example**



```
client.connections.get_datasource_type_id_by_name('cloudobjectstorage')

```





get\_datasource\_type\_uid\_by\_name(*name*)[[source]](_modules/ibm_watsonx_ai/connections.html#Connections.get_datasource_type_uid_by_name)[¶](#client.Connections.get_datasource_type_uid_by_name "Link to this definition")
Get a stored datasource type ID for the given datasource type name.


*Deprecated:* Use `Connections.get_datasource_type_id_by_name(name)` instead.



Parameters:
**name** (*str*) – name of datasource type



Returns:
ID of datasource type



Return type:
str




**Example**



```
client.connections.get_datasource_type_uid_by_name('cloudobjectstorage')

```





get\_details(*connection\_id=None*)[[source]](_modules/ibm_watsonx_ai/connections.html#Connections.get_details)[¶](#client.Connections.get_details "Link to this definition")
Get connection details for the given unique connection ID.
If no connection\_id is passed, details for all connections are returned.



Parameters:
**connection\_id** (*str*) – unique ID of the connection



Returns:
metadata of the stored connection



Return type:
dict




**Example**



```
connection_details = client.connections.get_details(connection_id)
connection_details = client.connections.get_details()

```





*static* get\_id(*connection\_details*)[[source]](_modules/ibm_watsonx_ai/connections.html#Connections.get_id)[¶](#client.Connections.get_id "Link to this definition")
Get ID of a stored connection.



Parameters:
**connection\_details** (*dict*) – metadata of the stored connection



Returns:
unique ID of the stored connection



Return type:
str




**Example**



```
connection_id = client.connection.get_id(connection_details)

```





*static* get\_uid(*connection\_details*)[[source]](_modules/ibm_watsonx_ai/connections.html#Connections.get_uid)[¶](#client.Connections.get_uid "Link to this definition")
Get the unique ID of a stored connection.


*Deprecated:* Use `Connections.get_id(details)` instead.



Parameters:
**connection\_details** (*dict*) – metadata of the stored connection



Returns:
unique ID of the stored connection



Return type:
str




**Example**



```
connection_uid = client.connection.get_uid(connection_details)

```





get\_uploaded\_db\_drivers()[[source]](_modules/ibm_watsonx_ai/connections.html#Connections.get_uploaded_db_drivers)[¶](#client.Connections.get_uploaded_db_drivers "Link to this definition")
Get uploaded db driver jar names and paths.
Supported for IBM Cloud Pak for Data, version 4.6.1 and up.


**Output**



Important


Returns dictionary containing name and path for connection files.


**return type**: Dict[Str, Str]



**Example**



```
>>> result = client.connections.get_uploaded_db_drivers()

```





list()[[source]](_modules/ibm_watsonx_ai/connections.html#Connections.list)[¶](#client.Connections.list "Link to this definition")
Return pd.DataFrame table with all stored connections in a table format.



Returns:
pandas.DataFrame with listed connections



Return type:
pandas.DataFrame




**Example**



```
client.connections.list()

```





list\_datasource\_types()[[source]](_modules/ibm_watsonx_ai/connections.html#Connections.list_datasource_types)[¶](#client.Connections.list_datasource_types "Link to this definition")
Print stored datasource types assets in a table format.



Returns:
pandas.DataFrame with listed datasource types



Return type:
pandas.DataFrame




**Example**
<https://test.cloud.ibm.com/apidocs/watsonx-ai#trainings-list>
.. code-block:: python



> client.connections.list\_datasource\_types()





list\_uploaded\_db\_drivers()[[source]](_modules/ibm_watsonx_ai/connections.html#Connections.list_uploaded_db_drivers)[¶](#client.Connections.list_uploaded_db_drivers "Link to this definition")
Return pd.DataFrame table with uploaded db driver jars in table a format. Supported for IBM Cloud Pak for Data only.



Returns:
pandas.DataFrame with listed uploaded db drivers



Return type:
pandas.DataFrame




**Example**



```
client.connections.list_uploaded_db_drivers()

```





sign\_db\_driver\_url(*jar\_name*)[[source]](_modules/ibm_watsonx_ai/connections.html#Connections.sign_db_driver_url)[¶](#client.Connections.sign_db_driver_url "Link to this definition")
Get a signed db driver jar URL to be used during JDBC generic connection creation.
The jar name passed as argument needs to be uploaded into the system first.
Supported for IBM Cloud Pak for Data only, version 4.0.4 and above.



Parameters:
**jar\_name** (*str*) – name of db driver jar



Returns:
URL of signed db driver



Return type:
str




**Example**



```
jar_uri = client.connections.sign_db_driver_url('db2jcc4.jar')

```





upload\_db\_driver(*path*)[[source]](_modules/ibm_watsonx_ai/connections.html#Connections.upload_db_driver)[¶](#client.Connections.upload_db_driver "Link to this definition")
Upload db driver jar. Supported for IBM Cloud Pak for Data only, version 4.0.4 and up.



Parameters:
**path** (*str*) – path to the db driver jar file




**Example**



```
client.connections.upload_db_driver('example/path/db2jcc4.jar')

```






*class* metanames.ConnectionMetaNames[[source]](_modules/metanames.html#ConnectionMetaNames)[¶](#metanames.ConnectionMetaNames "Link to this definition")
Set of MetaNames for Connection.


Available MetaNames:





| MetaName | Type | Required | Example value |
| --- | --- | --- | --- |
| NAME | str | Y | `my_space` |
| DESCRIPTION | str | N | `my_description` |
| DATASOURCE\_TYPE | str | Y | `1e3363a5-7ccf-4fff-8022-4850a8024b68` |
| PROPERTIES | dict | Y | `{'database': 'db_name', 'host': 'host_url', 'password': 'password', 'username': 'user'}` |
| FLAGS | list | N | `['personal_credentials']` |






Data assets[¶](#data-assets "Link to this heading")
---------------------------------------------------




*class* client.Assets(*client*)[[source]](_modules/ibm_watsonx_ai/assets.html#Assets)[¶](#client.Assets "Link to this definition")
Store and manage data assets.




ConfigurationMetaNames *= <ibm\_watsonx\_ai.metanames.AssetsMetaNames object>*[¶](#client.Assets.ConfigurationMetaNames "Link to this definition")
MetaNames for Data Assets creation.





create(*name*, *file\_path*)[[source]](_modules/ibm_watsonx_ai/assets.html#Assets.create)[¶](#client.Assets.create "Link to this definition")
Create a data asset and upload content to it.



Parameters:
* **name** (*str*) – name to be given to the data asset
* **file\_path** (*str*) – path to the content file to be uploaded



Returns:
metadata of the stored data asset



Return type:
dict




**Example**



```
asset_details = client.data_assets.create(name="sample_asset", file_path="/path/to/file")

```





delete(*asset\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/assets.html#Assets.delete)[¶](#client.Assets.delete "Link to this definition")
Delete a stored data asset.



Parameters:
**asset\_id** (*str*) – unique ID of the data asset



Returns:
status (“SUCCESS” or “FAILED”) or dictionary, if deleted asynchronously



Return type:
str or dict




**Example**



```
client.data_assets.delete(asset_id)

```





download(*asset\_id=None*, *filename=''*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/assets.html#Assets.download)[¶](#client.Assets.download "Link to this definition")
Download and store the content of a data asset.



Parameters:
* **asset\_id** (*str*) – unique ID of the data asset to be downloaded
* **filename** (*str*) – filename to be used for the downloaded file



Returns:
normalized path to the downloaded asset content



Return type:
str




**Example**



```
client.data_assets.download(asset_id,"sample_asset.csv")

```





get\_content(*asset\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/assets.html#Assets.get_content)[¶](#client.Assets.get_content "Link to this definition")
Download the content of a data asset.



Parameters:
**asset\_id** (*str*) – unique ID of the data asset to be downloaded



Returns:
the asset content



Return type:
bytes




**Example**



```
content = client.data_assets.get_content(asset_id).decode('ascii')

```





get\_details(*asset\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/assets.html#Assets.get_details)[¶](#client.Assets.get_details "Link to this definition")
Get data asset details. If no asset\_id is passed, details for all assets are returned.



Parameters:
**asset\_id** (*str*) – unique ID of the asset



Returns:
metadata of the stored data asset



Return type:
dict




**Example**



```
asset_details = client.data_assets.get_details(asset_id)

```





*static* get\_href(*asset\_details*)[[source]](_modules/ibm_watsonx_ai/assets.html#Assets.get_href)[¶](#client.Assets.get_href "Link to this definition")
Get the URL of a stored data asset.



Parameters:
**asset\_details** (*dict*) – details of the stored data asset



Returns:
href of the stored data asset



Return type:
str




**Example**



```
asset_details = client.data_assets.get_details(asset_id)
asset_href = client.data_assets.get_href(asset_details)

```





*static* get\_id(*asset\_details*)[[source]](_modules/ibm_watsonx_ai/assets.html#Assets.get_id)[¶](#client.Assets.get_id "Link to this definition")
Get the unique ID of a stored data asset.



Parameters:
**asset\_details** (*dict*) – details of the stored data asset



Returns:
unique ID of the stored data asset



Return type:
str




**Example**



```
asset_id = client.data_assets.get_id(asset_details)

```





list(*limit=None*)[[source]](_modules/ibm_watsonx_ai/assets.html#Assets.list)[¶](#client.Assets.list "Link to this definition")
Lists stored data assets in a table format.
If limit is set to none, only the first 50 records are shown.



Parameters:
**limit** (*int*) – limit number for fetched records



Return type:
DataFrame



Returns:
listed elements




**Example**



```
client.data_assets.list()

```





store(*meta\_props*)[[source]](_modules/ibm_watsonx_ai/assets.html#Assets.store)[¶](#client.Assets.store "Link to this definition")
Create a data asset and upload content to it.



Parameters:
**meta\_props** (*dict*) – metadata of the space configuration. To see available meta names, use:



```
client.data_assets.ConfigurationMetaNames.get()

```







**Example**


Example of data asset creation for files:



```
metadata = {
    client.data_assets.ConfigurationMetaNames.NAME: 'my data assets',
    client.data_assets.ConfigurationMetaNames.DESCRIPTION: 'sample description',
    client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: 'sample.csv'
}
asset_details = client.data_assets.store(meta_props=metadata)

```


Example of data asset creation using a connection:



```
metadata = {
    client.data_assets.ConfigurationMetaNames.NAME: 'my data assets',
    client.data_assets.ConfigurationMetaNames.DESCRIPTION: 'sample description',
    client.data_assets.ConfigurationMetaNames.CONNECTION_ID: '39eaa1ee-9aa4-4651-b8fe-95d3ddae',
    client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: 't1/sample.csv'
}
asset_details = client.data_assets.store(meta_props=metadata)

```


Example of data asset creation with a database sources type connection:



```
metadata = {
    client.data_assets.ConfigurationMetaNames.NAME: 'my data assets',
    client.data_assets.ConfigurationMetaNames.DESCRIPTION: 'sample description',
    client.data_assets.ConfigurationMetaNames.CONNECTION_ID: '23eaf1ee-96a4-4651-b8fe-95d3dadfe',
    client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: 't1'
}
asset_details = client.data_assets.store(meta_props=metadata)

```






*class* metanames.AssetsMetaNames[[source]](_modules/metanames.html#AssetsMetaNames)[¶](#metanames.AssetsMetaNames "Link to this definition")
Set of MetaNames for Data Asset Specs.


Available MetaNames:





| MetaName | Type | Required | Example value |
| --- | --- | --- | --- |
| NAME | str | Y | `my_data_asset` |
| DATA\_CONTENT\_NAME | str | Y | `/test/sample.csv` |
| CONNECTION\_ID | str | N | `39eaa1ee-9aa4-4651-b8fe-95d3ddae` |
| DESCRIPTION | str | N | `my_description` |






Deployments[¶](#deployments "Link to this heading")
---------------------------------------------------




*class* client.Deployments(*client*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments)[¶](#client.Deployments "Link to this definition")
Deploy and score published artifacts (models and functions).




create(*artifact\_id=None*, *meta\_props=None*, *rev\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.create)[¶](#client.Deployments.create "Link to this definition")
Create a deployment from an artifact. An artifact is a model or function that can be deployed.



Parameters:
* **artifact\_id** (*str*) – ID of the published artifact (the model or function ID)
* **meta\_props** (*dict**,* *optional*) – meta props. To see the available list of meta names, use:



```
client.deployments.ConfigurationMetaNames.get()

```
* **rev\_id** (*str**,* *optional*) – revision ID of the deployment



Returns:
metadata of the created deployment



Return type:
dict




**Example**



```
meta_props = {
    client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT NAME",
    client.deployments.ConfigurationMetaNames.ONLINE: {},
    client.deployments.ConfigurationMetaNames.HARDWARE_SPEC : { "id":  "e7ed1d6c-2e89-42d7-aed5-8sb972c1d2b"},
    client.deployments.ConfigurationMetaNames.SERVING_NAME : 'sample_deployment'
}
deployment_details = client.deployments.create(artifact_id, meta_props)

```





create\_job(*deployment\_id*, *meta\_props*, *retention=None*, *transaction\_id=None*, *\_asset\_id=None*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.create_job)[¶](#client.Deployments.create_job "Link to this definition")
Create an asynchronous deployment job.



Parameters:
* **deployment\_id** (*str*) – unique ID of the deployment
* **meta\_props** (*dict*) – metaprops. To see the available list of metanames,
use `client.deployments.ScoringMetaNames.get()`
or `client.deployments.DecisionOptimizationmetaNames.get()`
* **retention** (*int**,* *optional*) – how many job days job meta should be retained,
takes integer values >= -1, supported only on Cloud
* **transaction\_id** (*str**,* *optional*) – transaction ID to be passed with the payload



Returns:
metadata of the created async deployment job



Return type:
dict or str





Note


* The valid payloads for scoring input are either list of values, pandas or numpy dataframes.



**Example**



```
scoring_payload = {client.deployments.ScoringMetaNames.INPUT_DATA: [{'fields': ['GENDER','AGE','MARITAL_STATUS','PROFESSION'],
                                                                         'values': [['M',23,'Single','Student'],
                                                                                    ['M',55,'Single','Executive']]}]}
async_job = client.deployments.create_job(deployment_id, scoring_payload)

```





delete(*deployment\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.delete)[¶](#client.Deployments.delete "Link to this definition")
Delete deployment.



Parameters:
**deployment\_id** (*str*) – unique ID of the deployment



Returns:
status (“SUCCESS” or “FAILED”)



Return type:
str




**Example**



```
client.deployments.delete(deployment_id)

```





delete\_job(*job\_id=None*, *hard\_delete=False*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.delete_job)[¶](#client.Deployments.delete_job "Link to this definition")
Delete a deployment job that is running. This method can also delete metadata
details of completed or canceled jobs when hard\_delete parameter is set to True.



Parameters:
* **job\_id** (*str*) – unique ID of the deployment job to be deleted
* **hard\_delete** (*bool**,* *optional*) – specify True or False:


True - To delete the completed or canceled job.


False - To cancel the currently running deployment job.



Returns:
status (“SUCCESS” or “FAILED”)



Return type:
str




**Example**



```
client.deployments.delete_job(job_id)

```





generate(*deployment\_id*, *prompt=None*, *params=None*, *guardrails=False*, *guardrails\_hap\_params=None*, *guardrails\_pii\_params=None*, *concurrency\_limit=10*, *async\_mode=False*, *validate\_prompt\_variables=True*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.generate)[¶](#client.Deployments.generate "Link to this definition")
Generate a raw response with prompt for given deployment\_id.



Parameters:
* **deployment\_id** (*str*) – unique ID of the deployment
* **prompt** (*str**,* *optional*) – prompt needed for text generation. If deployment\_id points to the Prompt Template asset, then the prompt argument must be None, defaults to None
* **params** (*dict**,* *optional*) – meta props for text generation, use `ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()` to view the list of MetaNames
* **guardrails** (*bool**,* *optional*) – If True, then potentially hateful, abusive, and/or profane language (HAP) was detected
filter is toggle on for both prompt and generated text, defaults to False
* **guardrails\_hap\_params** (*dict**,* *optional*) – meta props for HAP moderations, use `ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()`
to view the list of MetaNames
* **concurrency\_limit** (*int**,* *optional*) – number of requests to be sent in parallel, maximum is 10
* **async\_mode** (*bool**,* *optional*) – If True, then yield results asynchronously (using generator). In this case both the prompt and
the generated text will be concatenated in the final response - under generated\_text, defaults
to False
* **validate\_prompt\_variables** (*bool*) – If True, prompt variables provided in params are validated with the ones in Prompt Template Asset.
This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True



Returns:
scoring result containing generated content



Return type:
dict







generate\_text(*deployment\_id*, *prompt=None*, *params=None*, *raw\_response=False*, *guardrails=False*, *guardrails\_hap\_params=None*, *guardrails\_pii\_params=None*, *concurrency\_limit=10*, *validate\_prompt\_variables=True*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.generate_text)[¶](#client.Deployments.generate_text "Link to this definition")
Given the selected deployment (deployment\_id), a text prompt as input, and the parameters and concurrency\_limit,
the selected inference will generate a completion text as generated\_text response.



Parameters:
* **deployment\_id** (*str*) – unique ID of the deployment
* **prompt** (*str**,* *optional*) – the prompt string or list of strings. If the list of strings is passed, requests will be managed in parallel with the rate of concurency\_limit, defaults to None
* **params** (*dict**,* *optional*) – meta props for text generation, use `ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()` to view the list of MetaNames
* **raw\_response** (*bool**,* *optional*) – returns the whole response object
* **guardrails** (*bool**,* *optional*) – If True, then potentially hateful, abusive, and/or profane language (HAP) was detected
filter is toggle on for both prompt and generated text, defaults to False
* **guardrails\_hap\_params** (*dict**,* *optional*) – meta props for HAP moderations, use `ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()`
to view the list of MetaNames
* **concurrency\_limit** (*int**,* *optional*) – number of requests to be sent in parallel, maximum is 10
* **validate\_prompt\_variables** (*bool*) – If True, prompt variables provided in params are validated with the ones in Prompt Template Asset.
This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True



Returns:
generated content



Return type:
str





Note


By default only the first occurance of HAPDetectionWarning is displayed. To enable printing all warnings of this category, use:



```
import warnings
from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

warnings.filterwarnings("always", category=HAPDetectionWarning)

```






generate\_text\_stream(*deployment\_id*, *prompt=None*, *params=None*, *raw\_response=False*, *guardrails=False*, *guardrails\_hap\_params=None*, *guardrails\_pii\_params=None*, *validate\_prompt\_variables=True*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.generate_text_stream)[¶](#client.Deployments.generate_text_stream "Link to this definition")
Given the selected deployment (deployment\_id), a text prompt as input and parameters,
the selected inference will generate a streamed text as generate\_text\_stream.



Parameters:
* **deployment\_id** (*str*) – unique ID of the deployment
* **prompt** (*str**,* *optional*) – the prompt string, defaults to None
* **params** (*dict**,* *optional*) – meta props for text generation, use `ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()` to view the list of MetaNames
* **raw\_response** (*bool**,* *optional*) – yields the whole response object
* **guardrails** (*bool**,* *optional*) – If True, then potentially hateful, abusive, and/or profane language (HAP) was detected
filter is toggle on for both prompt and generated text, defaults to False
* **guardrails\_hap\_params** (*dict**,* *optional*) – meta props for HAP moderations, use `ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()`
to view the list of MetaNames
* **validate\_prompt\_variables** (*bool*) – If True, prompt variables provided in params are validated with the ones in Prompt Template Asset.
This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True



Returns:
generated content



Return type:
str





Note


By default only the first occurance of HAPDetectionWarning is displayed. To enable printing all warnings of this category, use:



```
import warnings
from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

warnings.filterwarnings("always", category=HAPDetectionWarning)

```






get\_details(*deployment\_id=None*, *serving\_name=None*, *limit=None*, *asynchronous=False*, *get\_all=False*, *spec\_state=None*, *\_silent=False*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.get_details)[¶](#client.Deployments.get_details "Link to this definition")
Get information about deployment(s).
If deployment\_id is not passed, all deployment details are returned.



Parameters:
* **deployment\_id** (*str**,* *optional*) – unique ID of the deployment
* **serving\_name** (*str**,* *optional*) – serving name that filters deployments
* **limit** (*int**,* *optional*) – limit number of fetched records
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks
* **spec\_state** (*SpecStates**,* *optional*) – software specification state, can be used only when deployment\_id is None



Returns:
metadata of the deployment(s)



Return type:
dict (if deployment\_id is not None) or {“resources”: [dict]} (if deployment\_id is None)




**Example**



```
deployment_details = client.deployments.get_details(deployment_id)
deployment_details = client.deployments.get_details(deployment_id=deployment_id)
deployments_details = client.deployments.get_details()
deployments_details = client.deployments.get_details(limit=100)
deployments_details = client.deployments.get_details(limit=100, get_all=True)
deployments_details = []
for entry in client.deployments.get_details(limit=100, asynchronous=True, get_all=True):
    deployments_details.extend(entry)

```





get\_download\_url(*deployment\_details*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.get_download_url)[¶](#client.Deployments.get_download_url "Link to this definition")
Get deployment\_download\_url from the deployment details.



Parameters:
**deployment\_details** (*dict*) – created deployment details



Returns:
deployment download URL that is used to get file deployment (for example: Core ML)



Return type:
str




**Example**



```
deployment_url = client.deployments.get_download_url(deployment)

```





*static* get\_href(*deployment\_details*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.get_href)[¶](#client.Deployments.get_href "Link to this definition")
Get deployment\_href from the deployment details.



Parameters:
**deployment\_details** (*dict*) – metadata of the deployment



Returns:
deployment href that is used to manage the deployment



Return type:
str




**Example**



```
deployment_href = client.deployments.get_href(deployment)

```





*static* get\_id(*deployment\_details*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.get_id)[¶](#client.Deployments.get_id "Link to this definition")
Get the deployment ID from the deployment details.



Parameters:
**deployment\_details** (*dict*) – metadata of the deployment



Returns:
deployment ID that is used to manage the deployment



Return type:
str




**Example**



```
deployment_id = client.deployments.get_id(deployment)

```





get\_job\_details(*job\_id=None*, *include=None*, *limit=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.get_job_details)[¶](#client.Deployments.get_job_details "Link to this definition")
Get information about deployment job(s).
If deployment job\_id is not passed, all deployment jobs details are returned.



Parameters:
* **job\_id** (*str**,* *optional*) – unique ID of the job
* **include** (*str**,* *optional*) – fields to be retrieved from ‘decision\_optimization’
and ‘scoring’ section mentioned as value(s) (comma separated) as output response fields
* **limit** (*int**,* *optional*) – limit number of fetched records



Returns:
metadata of deployment job(s)



Return type:
dict (if job\_id is not None) or {“resources”: [dict]} (if job\_id is None)




**Example**



```
deployment_details = client.deployments.get_job_details()
deployments_details = client.deployments.get_job_details(job_id=job_id)

```





get\_job\_href(*job\_details*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.get_job_href)[¶](#client.Deployments.get_job_href "Link to this definition")
Get the href of a deployment job.



Parameters:
**job\_details** (*dict*) – metadata of the deployment job



Returns:
href of the deployment job



Return type:
str




**Example**



```
job_details = client.deployments.get_job_details(job_id=job_id)
job_status = client.deployments.get_job_href(job_details)

```





get\_job\_id(*job\_details*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.get_job_id)[¶](#client.Deployments.get_job_id "Link to this definition")
Get the unique ID of a deployment job.



Parameters:
**job\_details** (*dict*) – metadata of the deployment job



Returns:
unique ID of the deployment job



Return type:
str




**Example**



```
job_details = client.deployments.get_job_details(job_id=job_id)
job_status = client.deployments.get_job_id(job_details)

```





get\_job\_status(*job\_id*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.get_job_status)[¶](#client.Deployments.get_job_status "Link to this definition")
Get the status of a deployment job.



Parameters:
**job\_id** (*str*) – unique ID of the deployment job



Returns:
status of the deployment job



Return type:
dict




**Example**



```
job_status = client.deployments.get_job_status(job_id)

```





get\_job\_uid(*job\_details*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.get_job_uid)[¶](#client.Deployments.get_job_uid "Link to this definition")
Get the unique ID of a deployment job.


*Deprecated:* Use `get_job_id(job_details)` instead.



Parameters:
**job\_details** (*dict*) – metadata of the deployment job



Returns:
unique ID of the deployment job



Return type:
str




**Example**



```
job_details = client.deployments.get_job_details(job_uid=job_uid)
job_status = client.deployments.get_job_uid(job_details)

```





*static* get\_scoring\_href(*deployment\_details*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.get_scoring_href)[¶](#client.Deployments.get_scoring_href "Link to this definition")
Get scoring URL from deployment details.



Parameters:
**deployment\_details** (*dict*) – metadata of the deployment



Returns:
scoring endpoint URL that is used to make scoring requests



Return type:
str




**Example**



```
scoring_href = client.deployments.get_scoring_href(deployment)

```





*static* get\_serving\_href(*deployment\_details*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.get_serving_href)[¶](#client.Deployments.get_serving_href "Link to this definition")
Get serving URL from the deployment details.



Parameters:
**deployment\_details** (*dict*) – metadata of the deployment



Returns:
serving endpoint URL that is used to make scoring requests



Return type:
str




**Example**



```
scoring_href = client.deployments.get_serving_href(deployment)

```





*static* get\_uid(*deployment\_details*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.get_uid)[¶](#client.Deployments.get_uid "Link to this definition")
Get deployment\_uid from the deployment details.


*Deprecated:* Use `get_id(deployment_details)` instead.



Parameters:
**deployment\_details** (*dict*) – metadata of the deployment



Returns:
deployment UID that is used to manage the deployment



Return type:
str




**Example**



```
deployment_uid = client.deployments.get_uid(deployment)

```





is\_serving\_name\_available(*serving\_name*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.is_serving_name_available)[¶](#client.Deployments.is_serving_name_available "Link to this definition")
Check if the serving name is available for use.



Parameters:
**serving\_name** (*str*) – serving name that filters deployments



Returns:
information about whether the serving name is available



Return type:
bool




**Example**



```
is_available = client.deployments.is_serving_name_available('test')

```





list(*limit=None*, *artifact\_type=None*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.list)[¶](#client.Deployments.list "Link to this definition")
Returns deployments in a table format. If limit is set to None, only the first 50 records are shown.



Parameters:
* **limit** (*int**,* *optional*) – limit number of fetched records
* **artifact\_type** (*str**,* *optional*) – return only deployments with the specified artifact\_type



Returns:
pandas.DataFrame with the listed deployments



Return type:
pandas.DataFrame




**Example**



```
client.deployments.list()

```





list\_jobs(*limit=None*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.list_jobs)[¶](#client.Deployments.list_jobs "Link to this definition")
Return the async deployment jobs in a table format.
If the limit is set to None, only the first 50 records are shown.



Parameters:
**limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed deployment jobs



Return type:
pandas.DataFrame





Note


This method list only async deployment jobs created for WML deployment.



**Example**



```
client.deployments.list_jobs()

```





score(*deployment\_id*, *meta\_props*, *transaction\_id=None*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.score)[¶](#client.Deployments.score "Link to this definition")
Make scoring requests against the deployed artifact.



Parameters:
* **deployment\_id** (*str*) – unique ID of the deployment to be scored
* **meta\_props** (*dict*) – meta props for scoring, use `client.deployments.ScoringMetaNames.show()` to view the list of ScoringMetaNames
* **transaction\_id** (*str**,* *optional*) – transaction ID to be passed with the records during payload logging



Returns:
scoring result that contains prediction and probability



Return type:
dict





Note


* *client.deployments.ScoringMetaNames.INPUT\_DATA* is the only metaname valid for sync scoring.
* The valid payloads for scoring input are either list of values, pandas or numpy dataframes.



**Example**



```
scoring_payload = {client.deployments.ScoringMetaNames.INPUT_DATA:
    [{'fields':
        ['GENDER','AGE','MARITAL_STATUS','PROFESSION'],
        'values': [
            ['M',23,'Single','Student'],
            ['M',55,'Single','Executive']
        ]
    }]
}
predictions = client.deployments.score(deployment_id, scoring_payload)

```





update(*deployment\_id=None*, *changes=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/deployments.html#Deployments.update)[¶](#client.Deployments.update "Link to this definition")
Updates existing deployment metadata. If ASSET is patched, then ‘id’ field is mandatory
and it starts a deployment with the provided asset id/rev. Deployment ID remains the same.



Parameters:
* **deployment\_id** (*str*) – unique ID of deployment to be updated
* **changes** (*dict*) – elements to be changed, where keys are ConfigurationMetaNames



Returns:
metadata of the updated deployment



Return type:
dict or None




**Examples**



```
metadata = {client.deployments.ConfigurationMetaNames.NAME:"updated_Deployment"}
updated_deployment_details = client.deployments.update(deployment_id, changes=metadata)

metadata = {client.deployments.ConfigurationMetaNames.ASSET: {  "id": "ca0cd864-4582-4732-b365-3165598dc945",
                                                                "rev":"2" }}
deployment_details = client.deployments.update(deployment_id, changes=metadata)

```






*class* metanames.DeploymentMetaNames[[source]](_modules/metanames.html#DeploymentMetaNames)[¶](#metanames.DeploymentMetaNames "Link to this definition")
Set of MetaNames for Deployments Specs.


Available MetaNames:





| MetaName | Type | Required | Schema | Example value |
| --- | --- | --- | --- | --- |
| TAGS | list | N | `['string']` | `['string1', 'string2']` |
| NAME | str | N |  | `my_deployment` |
| DESCRIPTION | str | N |  | `my_deployment` |
| CUSTOM | dict | N |  | `{}` |
| ASSET | dict | N |  | `{'id': '4cedab6d-e8e4-4214-b81a-2ddb122db2ab', 'rev': '1'}` |
| PROMPT\_TEMPLATE | dict | N |  | `{'id': '4cedab6d-e8e4-4214-b81a-2ddb122db2ab'}` |
| HARDWARE\_SPEC | dict | N |  | `{'id': '3342-1ce536-20dc-4444-aac7-7284cf3befc'}` |
| HYBRID\_PIPELINE\_HARDWARE\_SPECS | list | N |  | `[{'node_runtime_id': 'auto_ai.kb', 'hardware_spec': {'id': '3342-1ce536-20dc-4444-aac7-7284cf3befc', 'num_nodes': '2'}}]` |
| ONLINE | dict | N |  | `{}` |
| BATCH | dict | N |  | `{}` |
| DETACHED | dict | N |  | `{}` |
| R\_SHINY | dict | N |  | `{'authentication': 'anyone_with_url'}` |
| VIRTUAL | dict | N |  | `{}` |
| OWNER | str | N |  | `<owner_id>` |
| BASE\_MODEL\_ID | str | N |  | `google/flan-ul2` |
| BASE\_DEPLOYMENT\_ID | str | N |  | `76a60161-facb-4968-a475-a6f1447c44bf` |
| PROMPT\_VARIABLES | dict | N |  | `{'key': 'value'}` |






*class* ibm\_watsonx\_ai.utils.enums.RShinyAuthenticationValues(*value*)[[source]](_modules/ibm_watsonx_ai/utils/enums.html#RShinyAuthenticationValues)[¶](#ibm_watsonx_ai.utils.enums.RShinyAuthenticationValues "Link to this definition")
Allowable values of R\_Shiny authentication.




ANYONE\_WITH\_URL *= 'anyone\_with\_url'*[¶](#ibm_watsonx_ai.utils.enums.RShinyAuthenticationValues.ANYONE_WITH_URL "Link to this definition")



ANY\_VALID\_USER *= 'any\_valid\_user'*[¶](#ibm_watsonx_ai.utils.enums.RShinyAuthenticationValues.ANY_VALID_USER "Link to this definition")



MEMBERS\_OF\_DEPLOYMENT\_SPACE *= 'members\_of\_deployment\_space'*[¶](#ibm_watsonx_ai.utils.enums.RShinyAuthenticationValues.MEMBERS_OF_DEPLOYMENT_SPACE "Link to this definition")




*class* metanames.ScoringMetaNames[[source]](_modules/metanames.html#ScoringMetaNames)[¶](#metanames.ScoringMetaNames "Link to this definition")
Set of MetaNames for Scoring.


Available MetaNames:





| MetaName | Type | Required | Schema | Example value |
| --- | --- | --- | --- | --- |
| NAME | str | N |  | `jobs test` |
| INPUT\_DATA | list | N | `[{'name(optional)': 'string', 'id(optional)': 'string', 'fields(optional)': 'array[string]', 'values': 'array[array[string]]'}]` | `[{'fields': ['name', 'age', 'occupation'], 'values': [['john', 23, 'student']]}]` |
| INPUT\_DATA\_REFERENCES | list | N | `[{'id(optional)': 'string', 'type(required)': 'string', 'connection(required)': {'href(required)': 'string'}, 'location(required)': {'bucket': 'string', 'path': 'string'}, 'schema(optional)': {'id(required)': 'string', 'fields(required)': [{'name(required)': 'string', 'type(required)': 'string', 'nullable(optional)': 'string'}]}}]` |  |
| OUTPUT\_DATA\_REFERENCE | dict | N | `{'type(required)': 'string', 'connection(required)': {'href(required)': 'string'}, 'location(required)': {'bucket': 'string', 'path': 'string'}, 'schema(optional)': {'id(required)': 'string', 'fields(required)': [{'name(required)': 'string', 'type(required)': 'string', 'nullable(optional)': 'string'}]}}` |  |
| EVALUATIONS\_SPEC | list | N | `[{'id(optional)': 'string', 'input_target(optional)': 'string', 'metrics_names(optional)': 'array[string]'}]` | `[{'id': 'string', 'input_target': 'string', 'metrics_names': ['auroc', 'accuracy']}]` |
| ENVIRONMENT\_VARIABLES | dict | N |  | `{'my_env_var1': 'env_var_value1', 'my_env_var2': 'env_var_value2'}` |
| SCORING\_PARAMETERS | dict | N |  | `{'forecast_window': 50}` |






*class* metanames.DecisionOptimizationMetaNames[[source]](_modules/metanames.html#DecisionOptimizationMetaNames)[¶](#metanames.DecisionOptimizationMetaNames "Link to this definition")
Set of MetaNames for Decision Optimization.


Available MetaNames:





| MetaName | Type | Required | Schema | Example value |
| --- | --- | --- | --- | --- |
| INPUT\_DATA | list | N | `[{'name(optional)': 'string', 'id(optional)': 'string', 'fields(optional)': 'array[string]', 'values': 'array[array[string]]'}]` | `[{'fields': ['name', 'age', 'occupation'], 'values': [['john', 23, 'student']]}]` |
| INPUT\_DATA\_REFERENCES | list | N | `[{'name(optional)': 'string', 'id(optional)': 'string', 'fields(optional)': 'array[string]', 'values': 'array[array[string]]'}]` | `[{'fields': ['name', 'age', 'occupation'], 'values': [['john', 23, 'student']]}]` |
| OUTPUT\_DATA | list | N | `[{'name(optional)': 'string'}]` |  |
| OUTPUT\_DATA\_REFERENCES | list | N | `{'name(optional)': 'string', 'type(required)': 'string', 'connection(required)': {'endpoint_url(required)': 'string', 'access_key_id(required)': 'string', 'secret_access_key(required)': 'string'}, 'location(required)': {'bucket': 'string', 'path': 'string'}, 'schema(optional)': {'id(required)': 'string', 'fields(required)': [{'name(required)': 'string', 'type(required)': 'string', 'nullable(optional)': 'string'}]}}` |  |
| SOLVE\_PARAMETERS | dict | N |  |  |






Export/Import[¶](#export-import "Link to this heading")
-------------------------------------------------------




*class* client.Export(*client*)[[source]](_modules/ibm_watsonx_ai/export_assets.html#Export)[¶](#client.Export "Link to this definition")


cancel(*export\_id*, *space\_id=None*, *project\_id=None*)[[source]](_modules/ibm_watsonx_ai/export_assets.html#Export.cancel)[¶](#client.Export.cancel "Link to this definition")
Cancel an export job. space\_id or project\_id has to be provided.



Note


To delete an export\_id job, use `delete()` API.




Parameters:
* **export\_id** (*str*) – export job identifier
* **space\_id** (*str**,* *optional*) – space identifier
* **project\_id** (*str**,* *optional*) – project identifier



Returns:
status (“SUCCESS” or “FAILED”)



Return type:
str




**Example**



```
client.export_assets.cancel(export_id='6213cf1-252f-424b-b52d-5cdd9814956c',
                            space_id='3421cf1-252f-424b-b52d-5cdd981495fe')

```





delete(*export\_id*, *space\_id=None*, *project\_id=None*)[[source]](_modules/ibm_watsonx_ai/export_assets.html#Export.delete)[¶](#client.Export.delete "Link to this definition")
Delete the given export\_id job. space\_id or project\_id has to be provided.



Parameters:
* **export\_id** (*str*) – export job identifier
* **space\_id** (*str**,* *optional*) – space identifier
* **project\_id** (*str**,* *optional*) – project identifier



Returns:
status (“SUCCESS” or “FAILED”)



Return type:
str




**Example**



```
client.export_assets.delete(export_id='6213cf1-252f-424b-b52d-5cdd9814956c',
                            space_id= '98a53931-a8c0-4c2f-8319-c793155e4598')

```





get\_details(*export\_id=None*, *space\_id=None*, *project\_id=None*, *limit=None*, *asynchronous=False*, *get\_all=False*)[[source]](_modules/ibm_watsonx_ai/export_assets.html#Export.get_details)[¶](#client.Export.get_details "Link to this definition")
Get metadata of a given export job. If no export\_id is specified, all export metadata is returned.



Parameters:
* **export\_id** (*str**,* *optional*) – export job identifier
* **space\_id** (*str**,* *optional*) – space identifier
* **project\_id** (*str**,* *optional*) – project identifier
* **limit** (*int**,* *optional*) – limit number of fetched records
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks



Returns:
export metadata



Return type:
dict (if export\_id is not None) or {“resources”: [dict]} (if export\_id is None)




**Example**



```
details = client.export_assets.get_details(export_id, space_id= '98a53931-a8c0-4c2f-8319-c793155e4598')
details = client.export_assets.get_details()
details = client.export_assets.get_details(limit=100)
details = client.export_assets.get_details(limit=100, get_all=True)
details = []
for entry in client.export_assets.get_details(limit=100, asynchronous=True, get_all=True):
    details.extend(entry)

```





get\_exported\_content(*export\_id*, *space\_id=None*, *project\_id=None*, *file\_path=None*)[[source]](_modules/ibm_watsonx_ai/export_assets.html#Export.get_exported_content)[¶](#client.Export.get_exported_content "Link to this definition")
Get the exported content as a zip file.



Parameters:
* **export\_id** (*str*) – export job identifier
* **space\_id** (*str**,* *optional*) – space identifier
* **project\_id** (*str**,* *optional*) – project identifier
* **file\_path** (*str**,* *optional*) – name of local file to create, this should be absolute path of the file
and the file shouldn’t exist



Returns:
path to the downloaded function content



Return type:
str




**Example**



```
client.exports.get_exported_content(export_id,
                                    space_id='98a53931-a8c0-4c2f-8319-c793155e4598',
                                    file_path='/home/user/my_exported_content.zip')

```





*static* get\_id(*export\_details*)[[source]](_modules/ibm_watsonx_ai/export_assets.html#Export.get_id)[¶](#client.Export.get_id "Link to this definition")
Get the ID of the export job from export details.



Parameters:
**export\_details** (*dict*) – metadata of the export job



Returns:
ID of the export job



Return type:
str




**Example**



```
id = client.export_assets.get_id(export_details)

```





list(*space\_id=None*, *project\_id=None*, *limit=None*)[[source]](_modules/ibm_watsonx_ai/export_assets.html#Export.list)[¶](#client.Export.list "Link to this definition")
Return export jobs in a table format. If limit is set to None, only the first 50 records are shown.



Parameters:
* **space\_id** (*str**,* *optional*) – space identifier
* **project\_id** (*str**,* *optional*) – project identifier
* **limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed connections



Return type:
pandas.DataFrame




**Example**



```
client.export_assets.list()

```





start(*meta\_props*, *space\_id=None*, *project\_id=None*)[[source]](_modules/ibm_watsonx_ai/export_assets.html#Export.start)[¶](#client.Export.start "Link to this definition")
Start the export. You must provide the space\_id or the project\_id.
ALL\_ASSETS is by default False. You don’t need to provide it unless it is set to True.
You must provide one of the following in the meta\_props: ALL\_ASSETS, ASSET\_TYPES, or ASSET\_IDS. Only one of these can be
provided.


In the meta\_props:


ALL\_ASSETS is a boolean. When set to True, it exports all assets in the given space.
ASSET\_IDS is an array that contains the list of assets IDs to be exported.
ASSET\_TYPES is used to provide the asset types to be exported. All assets of that asset type will be exported.



> Eg: wml\_model, wml\_model\_definition, wml\_pipeline, wml\_function, wml\_experiment,
> software\_specification, hardware\_specification, package\_extension, script



Parameters:
* **meta\_props** (*dict*) – metadata,
to see available meta names use `client.export_assets.ConfigurationMetaNames.get()`
* **space\_id** (*str**,* *optional*) – space identifier
* **project\_id** – project identifier



Returns:
Response json



Return type:
dict




**Example**



```
metadata = {
    client.export_assets.ConfigurationMetaNames.NAME: "export_model",
    client.export_assets.ConfigurationMetaNames.ASSET_IDS: ["13a53931-a8c0-4c2f-8319-c793155e7517",
                                                            "13a53931-a8c0-4c2f-8319-c793155e7518"]}

details = client.export_assets.start(meta_props=metadata, space_id="98a53931-a8c0-4c2f-8319-c793155e4598")

```



```
metadata = {
    client.export_assets.ConfigurationMetaNames.NAME: "export_model",
    client.export_assets.ConfigurationMetaNames.ASSET_TYPES: ["wml_model"]}

details = client.export_assets.start(meta_props=metadata, space_id="98a53931-a8c0-4c2f-8319-c793155e4598")

```



```
metadata = {
    client.export_assets.ConfigurationMetaNames.NAME: "export_model",
    client.export_assets.ConfigurationMetaNames.ALL_ASSETS: True}

details = client.export_assets.start(meta_props=metadata, space_id="98a53931-a8c0-4c2f-8319-c793155e4598")

```






*class* client.Import(*client*)[[source]](_modules/ibm_watsonx_ai/import_assets.html#Import)[¶](#client.Import "Link to this definition")


cancel(*import\_id*, *space\_id=None*, *project\_id=None*)[[source]](_modules/ibm_watsonx_ai/import_assets.html#Import.cancel)[¶](#client.Import.cancel "Link to this definition")
Cancel an import job. You must provide the space\_id or the project\_id.



Note


To delete an import\_id job, use delete() api




Parameters:
* **import\_id** (*str*) – import the job identifier
* **space\_id** (*str**,* *optional*) – space identifier
* **project\_id** (*str**,* *optional*) – project identifier




**Example**



```
client.import_assets.cancel(import_id='6213cf1-252f-424b-b52d-5cdd9814956c',
                            space_id='3421cf1-252f-424b-b52d-5cdd981495fe')

```





delete(*import\_id*, *space\_id=None*, *project\_id=None*)[[source]](_modules/ibm_watsonx_ai/import_assets.html#Import.delete)[¶](#client.Import.delete "Link to this definition")
Deletes the given import\_id job. You must provide the space\_id or the project\_id.



Parameters:
* **import\_id** (*str*) – import the job identifier
* **space\_id** (*str**,* *optional*) – space identifier
* **project\_id** (*str**,* *optional*) – project identifier




**Example**



```
client.import_assets.delete(import_id='6213cf1-252f-424b-b52d-5cdd9814956c',
                            space_id= '98a53931-a8c0-4c2f-8319-c793155e4598')

```





get\_details(*import\_id=None*, *space\_id=None*, *project\_id=None*, *limit=None*, *asynchronous=False*, *get\_all=False*)[[source]](_modules/ibm_watsonx_ai/import_assets.html#Import.get_details)[¶](#client.Import.get_details "Link to this definition")
Get metadata of the given import job. If no import\_id is specified, all import metadata is returned.



Parameters:
* **import\_id** (*str**,* *optional*) – import the job identifier
* **space\_id** (*str**,* *optional*) – space identifier
* **project\_id** (*str**,* *optional*) – project identifier
* **limit** (*int**,* *optional*) – limit number of fetched records
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks



Returns:
import(s) metadata



Return type:
dict (if import\_id is not None) or {“resources”: [dict]} (if import\_id is None)




**Example**



```
details = client.import_assets.get_details(import_id)
details = client.import_assets.get_details()
details = client.import_assets.get_details(limit=100)
details = client.import_assets.get_details(limit=100, get_all=True)
details = []
for entry in client.import_assets.get_details(limit=100, asynchronous=True, get_all=True):
    details.extend(entry)

```





*static* get\_id(*import\_details*)[[source]](_modules/ibm_watsonx_ai/import_assets.html#Import.get_id)[¶](#client.Import.get_id "Link to this definition")
Get ID of the import job from import details.



Parameters:
**import\_details** (*dict*) – metadata of the import job



Returns:
ID of the import job



Return type:
str




**Example**



```
id = client.import_assets.get_id(import_details)

```





list(*space\_id=None*, *project\_id=None*, *limit=None*)[[source]](_modules/ibm_watsonx_ai/import_assets.html#Import.list)[¶](#client.Import.list "Link to this definition")
Return import jobs in a table format. If limit is set to None, only the first 50 records are shown.



Parameters:
* **space\_id** (*str**,* *optional*) – space identifier
* **project\_id** (*str**,* *optional*) – project identifier
* **limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed assets



Return type:
pandas.DataFrame




**Example**



```
client.import_assets.list()

```





start(*file\_path*, *space\_id=None*, *project\_id=None*)[[source]](_modules/ibm_watsonx_ai/import_assets.html#Import.start)[¶](#client.Import.start "Link to this definition")
Start the import. You must provide the space\_id or the project\_id.



Parameters:
* **file\_path** (*str*) – file path to the zip file with exported assets
* **space\_id** (*str**,* *optional*) – space identifier
* **project\_id** (*str**,* *optional*) – project identifier



Returns:
response json



Return type:
dict




**Example**



```
details = client.import_assets.start(space_id="98a53931-a8c0-4c2f-8319-c793155e4598",
                                     file_path="/home/user/data_to_be_imported.zip")

```






Factsheets (IBM Cloud only)[¶](#factsheets-ibm-cloud-only "Link to this heading")
---------------------------------------------------------------------------------


**Warning!** Not supported for IBM Cloud Pak for Data.




*class* client.Factsheets(*client*)[[source]](_modules/ibm_watsonx_ai/factsheets.html#Factsheets)[¶](#client.Factsheets "Link to this definition")
Link WML Model to Model Entry.




list\_model\_entries(*catalog\_id=None*)[[source]](_modules/ibm_watsonx_ai/factsheets.html#Factsheets.list_model_entries)[¶](#client.Factsheets.list_model_entries "Link to this definition")
Return all WKC Model Entry assets for a catalog.



Parameters:
**catalog\_id** (*str**,* *optional*) – catalog ID where you want to register model. If no catalog\_id is provided, WKC Model Entry assets from all catalogs are listed.



Returns:
all WKC Model Entry assets for a catalog



Return type:
dict




**Example**



```
model_entries = client.factsheets.list_model_entries(catalog_id)

```





register\_model\_entry(*model\_id*, *meta\_props*, *catalog\_id=None*)[[source]](_modules/ibm_watsonx_ai/factsheets.html#Factsheets.register_model_entry)[¶](#client.Factsheets.register_model_entry "Link to this definition")
Link WML Model to Model Entry



Parameters:
* **model\_id** (*str*) – ID of the published model/asset
* **meta\_props** (*dict**[**str**,* *str**]*) – metaprops, to see the available list of meta names use:



```
client.factsheets.ConfigurationMetaNames.get()

```
* **catalog\_id** (*str**,* *optional*) – catalog ID where you want to register model



Returns:
metadata of the registration



Return type:
dict




**Example**



```
meta_props = {
    client.factsheets.ConfigurationMetaNames.ASSET_ID: '83a53931-a8c0-4c2f-8319-c793155e7517'}

registration_details = client.factsheets.register_model_entry(model_id, catalog_id, meta_props)

```


or



```
meta_props = {
    client.factsheets.ConfigurationMetaNames.NAME: "New model entry",
    client.factsheets.ConfigurationMetaNames.DESCRIPTION: "New model entry"}

registration_details = client.factsheets.register_model_entry(model_id, meta_props)

```





unregister\_model\_entry(*asset\_id*, *catalog\_id=None*)[[source]](_modules/ibm_watsonx_ai/factsheets.html#Factsheets.unregister_model_entry)[¶](#client.Factsheets.unregister_model_entry "Link to this definition")
Unregister WKC Model Entry



Parameters:
* **asset\_id** (*str*) – ID of the WKC model entry
* **catalog\_id** (*str**,* *optional*) – catalog ID where the asset is stored, when not provided,
default client space or project will be taken




**Example**



```
model_entries = client.factsheets.unregister_model_entry(asset_id='83a53931-a8c0-4c2f-8319-c793155e7517',
                                                         catalog_id='34553931-a8c0-4c2f-8319-c793155e7517')

```


or



```
client.set.default_space('98f53931-a8c0-4c2f-8319-c793155e7517')
model_entries = client.factsheets.unregister_model_entry(asset_id='83a53931-a8c0-4c2f-8319-c793155e7517')

```






*class* metanames.FactsheetsMetaNames[[source]](_modules/metanames.html#FactsheetsMetaNames)[¶](#metanames.FactsheetsMetaNames "Link to this definition")
Set of MetaNames for Factsheets metanames.


Available MetaNames:





| MetaName | Type | Required | Example value |
| --- | --- | --- | --- |
| ASSET\_ID | str | N | `13a53931-a8c0-4c2f-8319-c793155e7517` |
| NAME | str | N | `New model entry` |
| DESCRIPTION | str | N | `New model entry` |
| MODEL\_ENTRY\_CATALOG\_ID | str | Y | `13a53931-a8c0-4c2f-8319-c793155e7517` |






Hardware specifications[¶](#hardware-specifications "Link to this heading")
---------------------------------------------------------------------------




*class* client.HwSpec(*client*)[[source]](_modules/ibm_watsonx_ai/hw_spec.html#HwSpec)[¶](#client.HwSpec "Link to this definition")
Store and manage hardware specs.




ConfigurationMetaNames *= <ibm\_watsonx\_ai.metanames.HwSpecMetaNames object>*[¶](#client.HwSpec.ConfigurationMetaNames "Link to this definition")
MetaNames for Hardware Specification.





delete(*hw\_spec\_id*)[[source]](_modules/ibm_watsonx_ai/hw_spec.html#HwSpec.delete)[¶](#client.HwSpec.delete "Link to this definition")
Delete a hardware specification.



Parameters:
**hw\_spec\_id** (*str*) – unique ID of the hardware specification to be deleted



Returns:
status (“SUCCESS” or “FAILED”)



Return type:
str







get\_details(*hw\_spec\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/hw_spec.html#HwSpec.get_details)[¶](#client.HwSpec.get_details "Link to this definition")
Get hardware specification details.



Parameters:
**hw\_spec\_id** (*str*) – unique ID of the hardware spec



Returns:
metadata of the hardware specifications



Return type:
dict




**Example**



```
hw_spec_details = client.hardware_specifications.get_details(hw_spec_uid)

```





*static* get\_href(*hw\_spec\_details*)[[source]](_modules/ibm_watsonx_ai/hw_spec.html#HwSpec.get_href)[¶](#client.HwSpec.get_href "Link to this definition")
Get the URL of hardware specifications.



Parameters:
**hw\_spec\_details** (*dict*) – details of the hardware specifications



Returns:
href of the hardware specifications



Return type:
str




**Example**



```
hw_spec_details = client.hw_spec.get_details(hw_spec_id)
hw_spec_href = client.hw_spec.get_href(hw_spec_details)

```





*static* get\_id(*hw\_spec\_details*)[[source]](_modules/ibm_watsonx_ai/hw_spec.html#HwSpec.get_id)[¶](#client.HwSpec.get_id "Link to this definition")
Get the ID of a hardware specifications asset.



Parameters:
**hw\_spec\_details** (*dict*) – metadata of the hardware specifications



Returns:
unique ID of the hardware specifications



Return type:
str




**Example**



```
asset_id = client.hardware_specifications.get_id(hw_spec_details)

```





get\_id\_by\_name(*hw\_spec\_name*)[[source]](_modules/ibm_watsonx_ai/hw_spec.html#HwSpec.get_id_by_name)[¶](#client.HwSpec.get_id_by_name "Link to this definition")
Get the unique ID of a hardware specification for the given name.



Parameters:
**hw\_spec\_name** (*str*) – name of the hardware specification



Returns:
unique ID of the hardware specification



Return type:
str




**Example**



```
asset_id = client.hardware_specifications.get_id_by_name(hw_spec_name)

```





*static* get\_uid(*hw\_spec\_details*)[[source]](_modules/ibm_watsonx_ai/hw_spec.html#HwSpec.get_uid)[¶](#client.HwSpec.get_uid "Link to this definition")
Get the UID of a hardware specifications asset.


*Deprecated:* Use `get_id(hw_spec_details)` instead.



Parameters:
**hw\_spec\_details** (*dict*) – metadata of the hardware specifications



Returns:
unique ID of the hardware specifications



Return type:
str




**Example**



```
asset_uid = client.hardware_specifications.get_uid(hw_spec_details)

```





get\_uid\_by\_name(*hw\_spec\_name*)[[source]](_modules/ibm_watsonx_ai/hw_spec.html#HwSpec.get_uid_by_name)[¶](#client.HwSpec.get_uid_by_name "Link to this definition")
Get the unique ID of a hardware specification for the given name.


*Deprecated:* Use `get_id_by_name(hw_spec_name)` instead.



Parameters:
**hw\_spec\_name** (*str*) – name of the hardware specification



Returns:
unique ID of the hardware specification



Return type:
str




**Example**



```
asset_uid = client.hardware_specifications.get_uid_by_name(hw_spec_name)

```





list(*name=None*, *limit=None*)[[source]](_modules/ibm_watsonx_ai/hw_spec.html#HwSpec.list)[¶](#client.HwSpec.list "Link to this definition")
List hardware specifications in a table format.



Parameters:
* **name** (*str**,* *optional*) – unique ID of the hardware spec
* **limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed hardware specifications



Return type:
pandas.DataFrame




**Example**



```
client.hardware_specifications.list()

```





store(*meta\_props*)[[source]](_modules/ibm_watsonx_ai/hw_spec.html#HwSpec.store)[¶](#client.HwSpec.store "Link to this definition")
Create a hardware specification.



Parameters:
**meta\_props** (*dict*) – metadata of the hardware specification configuration. To see available meta names, use:



```
client.hardware_specifications.ConfigurationMetaNames.get()

```






Returns:
metadata of the created hardware specification



Return type:
dict




**Example**



```
meta_props = {
    client.hardware_specifications.ConfigurationMetaNames.NAME: "custom hardware specification",
    client.hardware_specifications.ConfigurationMetaNames.DESCRIPTION: "Custom hardware specification creted with SDK",
    client.hardware_specifications.ConfigurationMetaNames.NODES:{"cpu":{"units":"2"},"mem":{"size":"128Gi"},"gpu":{"num_gpu":1}}
 }

client.hardware_specifications.store(meta_props)

```






*class* metanames.HwSpecMetaNames[[source]](_modules/metanames.html#HwSpecMetaNames)[¶](#metanames.HwSpecMetaNames "Link to this definition")
Set of MetaNames for Hardware Specifications Specs.


Available MetaNames:





| MetaName | Type | Required | Example value |
| --- | --- | --- | --- |
| NAME | str | Y | `Custom Hardware Specification` |
| DESCRIPTION | str | N | `my_description` |
| NODES | dict | N | `{}` |
| SPARK | dict | N | `{}` |
| DATASTAGE | dict | N | `{}` |






Helpers[¶](#helpers "Link to this heading")
-------------------------------------------




*class* ibm\_watsonx\_ai.helpers.helpers.get\_credentials\_from\_config(*env\_name*, *credentials\_name*, *config\_path='./config.ini'*)[[source]](_modules/ibm_watsonx_ai/helpers/helpers.html#get_credentials_from_config)[¶](#ibm_watsonx_ai.helpers.helpers.get_credentials_from_config "Link to this definition")
Bases:


Load credentials from the config file.



```
[DEV_LC]

credentials = { }
cos_credentials = { }

```



Parameters:
* **env\_name** (*str*) – name of [ENV] defined in the config file
* **credentials\_name** (*str*) – name of credentials
* **config\_path** (*str*) – path to the config file



Returns:
loaded credentials



Return type:
dict




**Example**



```
get_credentials_from_config(env_name='DEV_LC', credentials_name='credentials')

```





Model definitions[¶](#model-definitions "Link to this heading")
---------------------------------------------------------------




*class* client.ModelDefinition(*client*)[[source]](_modules/ibm_watsonx_ai/model_definition.html#ModelDefinition)[¶](#client.ModelDefinition "Link to this definition")
Store and manage model definitions.




ConfigurationMetaNames *= <ibm\_watsonx\_ai.metanames.ModelDefinitionMetaNames object>*[¶](#client.ModelDefinition.ConfigurationMetaNames "Link to this definition")
MetaNames for model definition creation.





create\_revision(*model\_definition\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/model_definition.html#ModelDefinition.create_revision)[¶](#client.ModelDefinition.create_revision "Link to this definition")
Create a revision for the given model definition. Revisions are immutable once created.
The metadata and attachment of the model definition is taken and a revision is created out of it.



Parameters:
**model\_definition\_id** (*str*) – ID of the model definition



Returns:
revised metadata of the stored model definition



Return type:
dict




**Example**



```
model_definition_revision = client.model_definitions.create_revision(model_definition_id)

```





delete(*model\_definition\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/model_definition.html#ModelDefinition.delete)[¶](#client.ModelDefinition.delete "Link to this definition")
Delete a stored model definition.



Parameters:
**model\_definition\_id** (*str*) – unique ID of the stored model definition



Returns:
status (“SUCCESS” or “FAILED”)



Return type:
str




**Example**



```
client.model_definitions.delete(model_definition_id)

```





download(*model\_definition\_id*, *filename=None*, *rev\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/model_definition.html#ModelDefinition.download)[¶](#client.ModelDefinition.download "Link to this definition")
Download the content of a model definition asset.



Parameters:
* **model\_definition\_id** (*str*) – unique ID of the model definition asset to be downloaded
* **filename** (*str*) – filename to be used for the downloaded file
* **rev\_id** (*str**,* *optional*) – revision ID



Returns:
path to the downloaded asset content



Return type:
str




**Example**



```
client.model_definitions.download(model_definition_id, "model_definition_file")

```





get\_details(*model\_definition\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/model_definition.html#ModelDefinition.get_details)[¶](#client.ModelDefinition.get_details "Link to this definition")
Get metadata of a stored model definition. If no model\_definition\_id is passed,
details for all model definitions are returned.



Parameters:
**model\_definition\_id** (*str**,* *optional*) – unique ID of the model definition



Returns:
metadata of the model definition



Return type:
dict (if model\_definition\_id is not None)




**Example**





get\_href(*model\_definition\_details*)[[source]](_modules/ibm_watsonx_ai/model_definition.html#ModelDefinition.get_href)[¶](#client.ModelDefinition.get_href "Link to this definition")
Get the href of a stored model definition.



Parameters:
**model\_definition\_details** (*dict*) – details of the stored model definition



Returns:
href of the stored model definition



Return type:
str




**Example**



```
model_definition_id = client.model_definitions.get_href(model_definition_details)

```





get\_id(*model\_definition\_details*)[[source]](_modules/ibm_watsonx_ai/model_definition.html#ModelDefinition.get_id)[¶](#client.ModelDefinition.get_id "Link to this definition")
Get the unique ID of a stored model definition asset.



Parameters:
**model\_definition\_details** (*dict*) – metadata of the stored model definition asset



Returns:
unique ID of the stored model definition asset



Return type:
str




**Example**



```
asset_id = client.model_definition.get_id(asset_details)

```





get\_revision\_details(*model\_definition\_id=None*, *rev\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/model_definition.html#ModelDefinition.get_revision_details)[¶](#client.ModelDefinition.get_revision_details "Link to this definition")
Get metadata of a model definition.



Parameters:
* **model\_definition\_id** (*str*) – ID of the model definition
* **rev\_id** (*str**,* *optional*) – ID of the revision. If this parameter is not provided, it returns the latest revision. If there is no latest revision, it returns an error.



Returns:
metadata of the stored model definition



Return type:
dict




**Example**



```
script_details = client.model_definitions.get_revision_details(model_definition_id, rev_id)

```





get\_uid(*model\_definition\_details*)[[source]](_modules/ibm_watsonx_ai/model_definition.html#ModelDefinition.get_uid)[¶](#client.ModelDefinition.get_uid "Link to this definition")
Get the UID of the stored model.


*Deprecated:* Use `get_id(model_definition_details)` instead.



Parameters:
**model\_definition\_details** (*dict*) – details of the stored model definition



Returns:
UID of the stored model definition



Return type:
str




**Example**



```
model_definition_uid = client.model_definitions.get_uid(model_definition_details)

```





list(*limit=None*)[[source]](_modules/ibm_watsonx_ai/model_definition.html#ModelDefinition.list)[¶](#client.ModelDefinition.list "Link to this definition")
Return the stored model definition assets in a table format.
If limit is set to None, only the first 50 records are shown.



Parameters:
**limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed model definitions



Return type:
pandas.DataFrame




**Example**



```
client.model_definitions.list()

```





list\_revisions(*model\_definition\_id=None*, *limit=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/model_definition.html#ModelDefinition.list_revisions)[¶](#client.ModelDefinition.list_revisions "Link to this definition")
Return the stored model definition assets in a table format.
If limit is set to None, only the first 50 records are shown.



Parameters:
* **model\_definition\_id** (*str*) – unique ID of the model definition
* **limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed model definitions



Return type:
pandas.DataFrame




**Example**



```
client.model_definitions.list_revisions()

```





store(*model\_definition*, *meta\_props*)[[source]](_modules/ibm_watsonx_ai/model_definition.html#ModelDefinition.store)[¶](#client.ModelDefinition.store "Link to this definition")
Create a model definition.



Parameters:
* **meta\_props** (*dict*) – metadata of the model definition configuration. To see available meta names, use:



```
client.model_definitions.ConfigurationMetaNames.get()

```
* **model\_definition** (*str*) – path to the content file to be uploaded



Returns:
metadata of the created model definition



Return type:
dict




**Example**



```
client.model_definitions.store(model_definition, meta_props)

```





update(*model\_definition\_id*, *meta\_props=None*, *file\_path=None*)[[source]](_modules/ibm_watsonx_ai/model_definition.html#ModelDefinition.update)[¶](#client.ModelDefinition.update "Link to this definition")
Update the model definition with metadata, attachment, or both.



Parameters:
* **model\_definition\_id** (*str*) – ID of the model definition
* **meta\_props** (*dict*) – metadata of the model definition configuration to be updated
* **file\_path** (*str**,* *optional*) – path to the content file to be uploaded



Returns:
updated metadata of the model definition



Return type:
dict




**Example**



```
model_definition_details = client.model_definition.update(model_definition_id, meta_props, file_path)

```






*class* metanames.ModelDefinitionMetaNames[[source]](_modules/metanames.html#ModelDefinitionMetaNames)[¶](#metanames.ModelDefinitionMetaNames "Link to this definition")
Set of MetaNames for Model Definition.


Available MetaNames:





| MetaName | Type | Required | Schema | Example value |
| --- | --- | --- | --- | --- |
| NAME | str | Y |  | `my_model_definition` |
| DESCRIPTION | str | N |  | `my model_definition` |
| PLATFORM | dict | Y | `{'name(required)': 'string', 'versions(required)': ['versions']}` | `{'name': 'python', 'versions': ['3.10']}` |
| VERSION | str | Y |  | `1.0` |
| COMMAND | str | N |  | `python3 convolutional_network.py` |
| CUSTOM | dict | N |  | `{'field1': 'value1'}` |
| SPACE\_UID | str | N |  | `3c1ce536-20dc-426e-aac7-7284cf3befc6` |






Package extensions[¶](#package-extensions "Link to this heading")
-----------------------------------------------------------------




*class* client.PkgExtn(*client*)[[source]](_modules/ibm_watsonx_ai/pkg_extn.html#PkgExtn)[¶](#client.PkgExtn "Link to this definition")
Store and manage software Packages Extension specs.




ConfigurationMetaNames *= <ibm\_watsonx\_ai.metanames.PkgExtnMetaNames object>*[¶](#client.PkgExtn.ConfigurationMetaNames "Link to this definition")
MetaNames for Package Extensions creation.





delete(*pkg\_extn\_id*)[[source]](_modules/ibm_watsonx_ai/pkg_extn.html#PkgExtn.delete)[¶](#client.PkgExtn.delete "Link to this definition")
Delete a package extension.



Parameters:
**pkg\_extn\_id** (*str*) – unique ID of the package extension



Returns:
status (“SUCCESS” or “FAILED”) if deleted synchronously or dictionary with response



Return type:
str or dict




**Example**



```
client.package_extensions.delete(pkg_extn_id)

```





download(*pkg\_extn\_id*, *filename*)[[source]](_modules/ibm_watsonx_ai/pkg_extn.html#PkgExtn.download)[¶](#client.PkgExtn.download "Link to this definition")
Download a package extension.



Parameters:
* **pkg\_extn\_id** (*str*) – unique ID of the package extension to be downloaded
* **filename** (*str*) – filename to be used for the downloaded file



Returns:
path to the downloaded package extension content



Return type:
str




**Example**



```
client.package_extensions.download(pkg_extn_id,"sample_conda.yml/custom_library.zip")

```





get\_details(*pkg\_extn\_id*)[[source]](_modules/ibm_watsonx_ai/pkg_extn.html#PkgExtn.get_details)[¶](#client.PkgExtn.get_details "Link to this definition")
Get package extensions details.



Parameters:
**pkg\_extn\_id** (*str*) – unique ID of the package extension



Returns:
details of the package extension



Return type:
dict




**Example**



```
pkg_extn_details = client.pkg_extn.get_details(pkg_extn_id)

```





*static* get\_href(*pkg\_extn\_details*)[[source]](_modules/ibm_watsonx_ai/pkg_extn.html#PkgExtn.get_href)[¶](#client.PkgExtn.get_href "Link to this definition")
Get the URL of a stored package extension.



Parameters:
**pkg\_extn\_details** (*dict*) – details of the package extension



Returns:
href of the package extension



Return type:
str




**Example**



```
pkg_extn_details = client.package_extensions.get_details(pkg_extn_id)
pkg_extn_href = client.package_extensions.get_href(pkg_extn_details)

```





*static* get\_id(*pkg\_extn\_details*)[[source]](_modules/ibm_watsonx_ai/pkg_extn.html#PkgExtn.get_id)[¶](#client.PkgExtn.get_id "Link to this definition")
Get the unique ID of a package extension.



Parameters:
**pkg\_extn\_details** (*dict*) – details of the package extension



Returns:
unique ID of the package extension



Return type:
str




**Example**



```
asset_id = client.package_extensions.get_id(pkg_extn_details)

```





get\_id\_by\_name(*pkg\_extn\_name*)[[source]](_modules/ibm_watsonx_ai/pkg_extn.html#PkgExtn.get_id_by_name)[¶](#client.PkgExtn.get_id_by_name "Link to this definition")
Get the ID of a package extension.



Parameters:
**pkg\_extn\_name** (*str*) – name of the package extension



Returns:
unique ID of the package extension



Return type:
str




**Example**



```
asset_id = client.package_extensions.get_id_by_name(pkg_extn_name)

```





list()[[source]](_modules/ibm_watsonx_ai/pkg_extn.html#PkgExtn.list)[¶](#client.PkgExtn.list "Link to this definition")
List the package extensions in a table format.



Returns:
pandas.DataFrame with listed package extensions



Return type:
pandas.DataFrame





```
client.package_extensions.list()

```





store(*meta\_props*, *file\_path*)[[source]](_modules/ibm_watsonx_ai/pkg_extn.html#PkgExtn.store)[¶](#client.PkgExtn.store "Link to this definition")
Create a package extension.



Parameters:
* **meta\_props** (*dict*) – metadata of the package extension. To see available meta names, use:



```
client.package_extensions.ConfigurationMetaNames.get()

```
* **file\_path** (*str*) – path to the file to be uploaded as a package extension



Returns:
metadata of the package extension



Return type:
dict




**Example**



```
meta_props = {
    client.package_extensions.ConfigurationMetaNames.NAME: "skl_pipeline_heart_problem_prediction",
    client.package_extensions.ConfigurationMetaNames.DESCRIPTION: "description scikit-learn_0.20",
    client.package_extensions.ConfigurationMetaNames.TYPE: "conda_yml"
}

pkg_extn_details = client.package_extensions.store(meta_props=meta_props, file_path="/path/to/file")

```






*class* metanames.PkgExtnMetaNames[[source]](_modules/metanames.html#PkgExtnMetaNames)[¶](#metanames.PkgExtnMetaNames "Link to this definition")
Set of MetaNames for Package Extensions Specs.


Available MetaNames:





| MetaName | Type | Required | Example value |
| --- | --- | --- | --- |
| NAME | str | Y | `Python 3.10 with pre-installed ML package` |
| DESCRIPTION | str | N | `my_description` |
| TYPE | str | Y | `conda_yml/custom_library` |






Parameter Sets[¶](#parameter-sets "Link to this heading")
---------------------------------------------------------




*class* client.ParameterSets(*client*)[[source]](_modules/ibm_watsonx_ai/parameter_sets.html#ParameterSets)[¶](#client.ParameterSets "Link to this definition")
Store and manage parameter sets.




ConfigurationMetaNames *= <ibm\_watsonx\_ai.metanames.ParameterSetsMetaNames object>*[¶](#client.ParameterSets.ConfigurationMetaNames "Link to this definition")
MetaNames for Parameter Sets creation.





create(*meta\_props*)[[source]](_modules/ibm_watsonx_ai/parameter_sets.html#ParameterSets.create)[¶](#client.ParameterSets.create "Link to this definition")
Create a parameter set.



Parameters:
**meta\_props** (*dict*) – metadata of the space configuration. To see available meta names, use:



```
client.parameter_sets.ConfigurationMetaNames.get()

```






Returns:
metadata of the stored parameter set



Return type:
dict




**Example**



```
meta_props = {
    client.parameter_sets.ConfigurationMetaNames.NAME: "Example name",
    client.parameter_sets.ConfigurationMetaNames.DESCRIPTION: "Example description",
    client.parameter_sets.ConfigurationMetaNames.PARAMETERS: [
        {
            "name": "string",
            "description": "string",
            "prompt": "string",
            "type": "string",
            "subtype": "string",
            "value": "string",
            "valid_values": [
                "string"
            ]
        }
    ],
    client.parameter_sets.ConfigurationMetaNames.VALUE_SETS: [
        {
            "name": "string",
            "values": [
                {
                    "name": "string",
                    "value": "string"
                }
            ]
        }
    ]
}

parameter_sets_details = client.parameter_sets.create(meta_props)

```





delete(*parameter\_set\_id*)[[source]](_modules/ibm_watsonx_ai/parameter_sets.html#ParameterSets.delete)[¶](#client.ParameterSets.delete "Link to this definition")
Delete a parameter set.



Parameters:
**parameter\_set\_id** (*str*) – unique ID of the parameter set



Returns:
status (“SUCCESS” or “FAILED”)



Return type:
str




**Example**



```
client.parameter_sets.delete(parameter_set_id)

```





get\_details(*parameter\_set\_id=None*)[[source]](_modules/ibm_watsonx_ai/parameter_sets.html#ParameterSets.get_details)[¶](#client.ParameterSets.get_details "Link to this definition")
Get parameter set details. If no parameter\_sets\_id is passed, details for all parameter sets
are returned.



Parameters:
**parameter\_set\_id** (*str**,* *optional*) – ID of the software specification



Returns:
metadata of the stored parameter set(s)



Return type:
* **dict** - if parameter\_set\_id is not None
* **{“parameter\_sets”: [dict]}** - if parameter\_set\_id is None







**Examples**


If parameter\_set\_id is None:



```
parameter_sets_details = client.parameter_sets.get_details()

```


If parameter\_set\_id is given:



```
parameter_sets_details = client.parameter_sets.get_details(parameter_set_id)

```





get\_id\_by\_name(*parameter\_set\_name*)[[source]](_modules/ibm_watsonx_ai/parameter_sets.html#ParameterSets.get_id_by_name)[¶](#client.ParameterSets.get_id_by_name "Link to this definition")
Get the unique ID of a parameter set.



Parameters:
**parameter\_set\_name** (*str*) – name of the parameter set



Returns:
unique ID of the parameter set



Return type:
str




**Example**



```
asset_id = client.parameter_sets.get_id_by_name(parameter_set_name)

```





list(*limit=None*)[[source]](_modules/ibm_watsonx_ai/parameter_sets.html#ParameterSets.list)[¶](#client.ParameterSets.list "Link to this definition")
List parameter sets in a table format.



Parameters:
**limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed parameter sets



Return type:
pandas.DataFrame




**Example**



```
client.parameter_sets.list()

```





update(*parameter\_set\_id*, *new\_data*, *file\_path*)[[source]](_modules/ibm_watsonx_ai/parameter_sets.html#ParameterSets.update)[¶](#client.ParameterSets.update "Link to this definition")
Update parameter sets.



Parameters:
* **parameter\_set\_id** (*str*) – unique ID of the parameter sets
* **new\_data** (*str**,* *list*) – new data for parameters
* **file\_path** (*str*) – path to update



Returns:
metadata of the updated parameter sets



Return type:
dict




**Example for description**



```
new_description_data = "New description"
parameter_set_details = client.parameter_sets.update(parameter_set_id, new_description_data, "description")

```


**Example for parameters**



```
new_parameters_data = [
    {
        "name": "string",
        "description": "new_description",
        "prompt": "new_string",
        "type": "new_string",
        "subtype": "new_string",
        "value": "new_string",
        "valid_values": [
            "new_string"
        ]
    }
]
parameter_set_details = client.parameter_sets.update(parameter_set_id, new_parameters_data, "parameters")

```


**Example for value\_sets**



```
new_value_sets_data = [
    {
        "name": "string",
        "values": [
            {
                "name": "string",
                "value": "new_string"
            }
        ]
    }
]
parameter_set_details = client.parameter_sets.update_value_sets(parameter_set_id, new_value_sets_data, "value_sets")

```






*class* metanames.ParameterSetsMetaNames[[source]](_modules/metanames.html#ParameterSetsMetaNames)[¶](#metanames.ParameterSetsMetaNames "Link to this definition")
Set of MetaNames for Parameter Sets metanames.


Available MetaNames:





| MetaName | Type | Required | Example value |
| --- | --- | --- | --- |
| NAME | str | Y | `sample name` |
| DESCRIPTION | str | N | `sample description` |
| PARAMETERS | list | Y | `[{'name': 'string', 'description': 'string', 'prompt': 'string', 'type': 'string', 'subtype': 'string', 'value': 'string', 'valid_values': ['string']}]` |
| VALUE\_SETS | list | N | `[{'name': 'string', 'values': [{'name': 'string', 'value': 'string'}]}]` |






Repository[¶](#repository "Link to this heading")
-------------------------------------------------




*class* client.Repository(*client*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository)[¶](#client.Repository "Link to this definition")
Store and manage models, functions, spaces, pipelines, and experiments
using the Watson Machine Learning Repository.


To view ModelMetaNames, use:



```
client.repository.ModelMetaNames.show()

```


To view ExperimentMetaNames, use:



```
client.repository.ExperimentMetaNames.show()

```


To view FunctionMetaNames, use:



```
client.repository.FunctionMetaNames.show()

```


To view PipelineMetaNames, use:



```
client.repository.PipelineMetaNames.show()

```




*class* ModelAssetTypes(*DO\_DOCPLEX\_20\_1='do-docplex\_20.1'*, *DO\_OPL\_20\_1='do-opl\_20.1'*, *DO\_CPLEX\_20\_1='do-cplex\_20.1'*, *DO\_CPO\_20\_1='do-cpo\_20.1'*, *DO\_DOCPLEX\_22\_1='do-docplex\_22.1'*, *DO\_OPL\_22\_1='do-opl\_22.1'*, *DO\_CPLEX\_22\_1='do-cplex\_22.1'*, *DO\_CPO\_22\_1='do-cpo\_22.1'*, *WML\_HYBRID\_0\_1='wml-hybrid\_0.1'*, *PMML\_4\_2\_1='pmml\_4.2.1'*, *PYTORCH\_ONNX\_1\_12='pytorch-onnx\_1.12'*, *PYTORCH\_ONNX\_RT22\_2='pytorch-onnx\_rt22.2'*, *PYTORCH\_ONNX\_2\_0='pytorch-onnx\_2.0'*, *PYTORCH\_ONNX\_RT23\_1='pytorch-onnx\_rt23.1'*, *SCIKIT\_LEARN\_1\_1='scikit-learn\_1.1'*, *MLLIB\_3\_3='mllib\_3.3'*, *SPSS\_MODELER\_17\_1='spss-modeler\_17.1'*, *SPSS\_MODELER\_18\_1='spss-modeler\_18.1'*, *SPSS\_MODELER\_18\_2='spss-modeler\_18.2'*, *TENSORFLOW\_2\_9='tensorflow\_2.9'*, *TENSORFLOW\_RT22\_2='tensorflow\_rt22.2'*, *TENSORFLOW\_2\_12='tensorflow\_2.12'*, *TENSORFLOW\_RT23\_1='tensorflow\_rt23.1'*, *XGBOOST\_1\_6='xgboost\_1.6'*, *PROMPT\_TUNE\_1\_0='prompt\_tune\_1.0'*, *CUSTOM\_FOUNDATION\_MODEL\_1\_0='custom\_foundation\_model\_1.0'*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.ModelAssetTypes)[¶](#client.Repository.ModelAssetTypes "Link to this definition")
Data class with supported model asset types.





create\_experiment\_revision(*experiment\_id*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.create_experiment_revision)[¶](#client.Repository.create_experiment_revision "Link to this definition")
Create a new experiment revision.



Parameters:
**experiment\_id** (*str*) – unique ID of the stored experiment



Returns:
new revision details of the stored experiment



Return type:
dict




**Example**



```
experiment_revision_artifact = client.repository.create_experiment_revision(experiment_id)

```





create\_function\_revision(*function\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.create_function_revision)[¶](#client.Repository.create_function_revision "Link to this definition")
Create a new function revision.



Parameters:
**function\_id** (*str*) – unique ID of the function



Returns:
revised metadata of the stored function



Return type:
dict




**Example**



```
client.repository.create_function_revision(pipeline_id)

```





create\_model\_revision(*model\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.create_model_revision)[¶](#client.Repository.create_model_revision "Link to this definition")
Create a revision for a given model ID.



Parameters:
**model\_id** (*str*) – ID of the stored model



Returns:
revised metadata of the stored model



Return type:
dict




**Example**



```
model_details = client.repository.create_model_revision(model_id)

```





create\_pipeline\_revision(*pipeline\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.create_pipeline_revision)[¶](#client.Repository.create_pipeline_revision "Link to this definition")
Create a new pipeline revision.



Parameters:
**pipeline\_id** (*str*) – unique ID of the pipeline



Returns:
details of the pipeline revision



Return type:
dict




**Example**



```
client.repository.create_pipeline_revision(pipeline_id)

```





create\_revision(*artifact\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.create_revision)[¶](#client.Repository.create_revision "Link to this definition")
Create a revision for passed artifact\_id.



Parameters:
**artifact\_id** (*str*) – unique ID of a stored model, experiment, function, or pipelines



Returns:
artifact new revision metadata



Return type:
dict




**Example**



```
details = client.repository.create_revision(artifact_id)

```





delete(*artifact\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.delete)[¶](#client.Repository.delete "Link to this definition")
Delete a model, experiment, pipeline, or function from the repository.



Parameters:
**artifact\_id** (*str*) – unique ID of the stored model, experiment, function, or pipeline



Returns:
status “SUCCESS” if deletion is successful



Return type:
Literal[“SUCCESS”]




**Example**



```
client.repository.delete(artifact_id)

```





download(*artifact\_id=None*, *filename='downloaded\_artifact.tar.gz'*, *rev\_id=None*, *format=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.download)[¶](#client.Repository.download "Link to this definition")
Download the configuration file for an artifact with the specified ID.



Parameters:
* **artifact\_id** (*str*) – unique ID of the model or function
* **filename** (*str**,* *optional*) – name of the file to which the artifact content will be downloaded
* **rev\_id** (*str**,* *optional*) – revision ID
* **format** (*str**,* *optional*) – format of the content, applicable for models



Returns:
path to the downloaded artifact content



Return type:
str




**Examples**



```
client.repository.download(model_id, 'my_model.tar.gz')
client.repository.download(model_id, 'my_model.json') # if original model was saved as json, works only for xgboost 1.3

```





get\_details(*artifact\_id=None*, *spec\_state=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_details)[¶](#client.Repository.get_details "Link to this definition")
Get metadata of stored artifacts. If artifact\_id is not specified, the metadata of all models, experiments,
functions, and pipelines is returned.



Parameters:
* **artifact\_id** (*str**,* *optional*) – unique ID of the stored model, experiment, function, or pipeline
* **spec\_state** (*SpecStates**,* *optional*) – software specification state, can be used only when artifact\_id is None



Returns:
metadata of the stored artifact(s)



Return type:
dict (if artifact\_id is not None) or {“resources”: [dict]} (if artifact\_id is None)




**Examples**



```
details = client.repository.get_details(artifact_id)
details = client.repository.get_details()

```


Example of getting all repository assets with deprecated software specifications:



```
from ibm_watsonx_ai.lifecycle import SpecStates

details = client.repository.get_details(spec_state=SpecStates.DEPRECATED)

```





get\_experiment\_details(*experiment\_id=None*, *limit=None*, *asynchronous=False*, *get\_all=False*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_experiment_details)[¶](#client.Repository.get_experiment_details "Link to this definition")
Get metadata of the experiment(s). If no experiment ID is specified, all experiment metadata is returned.



Parameters:
* **experiment\_id** (*str**,* *optional*) – ID of the experiment
* **limit** (*int**,* *optional*) – limit number of fetched records
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks



Returns:
experiment metadata



Return type:
dict (if ID is not None) or {“resources”: [dict]} (if ID is None)




**Example**



```
experiment_details = client.repository.get_experiment_details(experiment_id)
experiment_details = client.repository.get_experiment_details()
experiment_details = client.repository.get_experiment_details(limit=100)
experiment_details = client.repository.get_experiment_details(limit=100, get_all=True)
experiment_details = []
for entry in client.repository.get_experiment_details(limit=100, asynchronous=True, get_all=True):
    experiment_details.extend(entry)

```





*static* get\_experiment\_href(*experiment\_details*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_experiment_href)[¶](#client.Repository.get_experiment_href "Link to this definition")
Get the href of a stored experiment.



Parameters:
**experiment\_details** (*dict*) – metadata of the stored experiment



Returns:
href of the stored experiment



Return type:
str




**Example**



```
experiment_details = client.repository.get_experiment_details(experiment_id)
experiment_href = client.repository.get_experiment_href(experiment_details)

```





*static* get\_experiment\_id(*experiment\_details*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_experiment_id)[¶](#client.Repository.get_experiment_id "Link to this definition")
Get the unique ID of a stored experiment.



Parameters:
**experiment\_details** (*dict*) – metadata of the stored experiment



Returns:
unique ID of the stored experiment



Return type:
str




**Example**



```
experiment_details = client.repository.get_experiment_details(experiment_id)
experiment_id = client.repository.get_experiment_id(experiment_details)

```





get\_experiment\_revision\_details(*experiment\_id*, *rev\_id*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_experiment_revision_details)[¶](#client.Repository.get_experiment_revision_details "Link to this definition")
Get metadata of a stored experiments revisions.



Parameters:
* **experiment\_id** (*str*) – ID of the stored experiment
* **rev\_id** (*str*) – rev\_id number of the stored experiment



Returns:
revision metadata of the stored experiment



Return type:
dict




Example:



```
experiment_details = client.repository.get_experiment_revision_details(experiment_id, rev_id)

```





get\_function\_details(*function\_id=None*, *limit=None*, *asynchronous=False*, *get\_all=False*, *spec\_state=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_function_details)[¶](#client.Repository.get_function_details "Link to this definition")
Get metadata of function(s). If no function ID is specified, the metadata of all functions is returned.



Parameters:
* **function\_id** – ID of the function
* **limit** (*int**,* *optional*) – limit number of fetched records
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks
* **spec\_state** (*SpecStates**,* *optional*) – software specification state, can be used only when model\_id is None



Type:
str, optional



Returns:
metadata of the function



Return type:
dict (if ID is not None) or {“resources”: [dict]} (if ID is None)





Note


In current implementation setting spec\_state=True may break set limit,
returning less records than stated by set limit.



**Examples**



```
function_details = client.repository.get_function_details(function_id)
function_details = client.repository.get_function_details()
function_details = client.repository.get_function_details(limit=100)
function_details = client.repository.get_function_details(limit=100, get_all=True)
function_details = []
for entry in client.repository.get_function_details(limit=100, asynchronous=True, get_all=True):
    function_details.extend(entry)

```





*static* get\_function\_href(*function\_details*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_function_href)[¶](#client.Repository.get_function_href "Link to this definition")
Get the URL of a stored function.



Parameters:
**function\_details** (*dict*) – details of the stored function



Returns:
href of the stored function



Return type:
str




**Example**



```
function_details = client.repository.get_function_details(function_id)
function_url = client.repository.get_function_href(function_details)

```





*static* get\_function\_id(*function\_details*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_function_id)[¶](#client.Repository.get_function_id "Link to this definition")
Get ID of stored function.



Parameters:
**function\_details** (*dict*) – metadata of the stored function



Returns:
ID of stored function



Return type:
str




**Example**



```
function_details = client.repository.get_function_details(function_id)
function_id = client.repository.get_function_id(function_details)

```





get\_function\_revision\_details(*function\_id*, *rev\_id*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_function_revision_details)[¶](#client.Repository.get_function_revision_details "Link to this definition")
Get metadata of a specific revision of a stored function.



Parameters:
* **function\_id** (*str*) – definition of the stored function
* **rev\_id** (*str*) – unique ID of the function revision



Returns:
stored function revision metadata



Return type:
dict




**Example**



```
function_revision_details = client.repository.get_function_revision_details(function_id, rev_id)

```





get\_model\_details(*model\_id=None*, *limit=None*, *asynchronous=False*, *get\_all=False*, *spec\_state=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_model_details)[¶](#client.Repository.get_model_details "Link to this definition")
Get metadata of stored models. If no model\_id is specified, the metadata of all models is returned.



Parameters:
* **model\_id** (*str**,* *optional*) – ID of the stored model, definition, or pipeline
* **limit** (*int**,* *optional*) – limit number of fetched records
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks
* **spec\_state** (*SpecStates**,* *optional*) – software specification state, can be used only when model\_id is None



Returns:
metadata of the stored model(s)



Return type:
dict (if ID is not None) or {“resources”: [dict]} (if ID is None)





Note


In current implementation setting spec\_state=True may break set limit,
returning less records than stated by set limit.



**Example**



```
model_details = client.repository.get_model_details(model_id)
models_details = client.repository.get_model_details()
models_details = client.repository.get_model_details(limit=100)
models_details = client.repository.get_model_details(limit=100, get_all=True)
models_details = []
for entry in client.repository.get_model_details(limit=100, asynchronous=True, get_all=True):
    models_details.extend(entry)

```





*static* get\_model\_href(*model\_details*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_model_href)[¶](#client.Repository.get_model_href "Link to this definition")
Get the URL of a stored model.



Parameters:
**model\_details** (*dict*) – details of the stored model



Returns:
URL of the stored model



Return type:
str




**Example**



```
model_url = client.repository.get_model_href(model_details)

```





*static* get\_model\_id(*model\_details*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_model_id)[¶](#client.Repository.get_model_id "Link to this definition")
Get the ID of a stored model.



Parameters:
**model\_details** (*dict*) – details of the stored model



Returns:
ID of the stored model



Return type:
str




**Example**



```
model_id = client.repository.get_model_id(model_details)

```





get\_model\_revision\_details(*model\_id=None*, *rev\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_model_revision_details)[¶](#client.Repository.get_model_revision_details "Link to this definition")
Get metadata of a stored model’s specific revision.



Parameters:
* **model\_id** (*str*) – ID of the stored model, definition, or pipeline
* **rev\_id** (*str*) – unique ID of the stored model revision



Returns:
metadata of the stored model(s)



Return type:
dict




**Example**



```
model_details = client.repository.get_model_revision_details(model_id, rev_id)

```





get\_pipeline\_details(*pipeline\_id=None*, *limit=None*, *asynchronous=False*, *get\_all=False*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_pipeline_details)[¶](#client.Repository.get_pipeline_details "Link to this definition")
Get metadata of stored pipeline(s). If pipeline ID is not specified, the metadata of all pipelines is returned.



Parameters:
* **pipeline\_id** (*str**,* *optional*) – ID of the pipeline
* **limit** (*int**,* *optional*) – limit number of fetched records
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks



Returns:
metadata of pipeline(s)



Return type:
dict (if ID is not None) or {“resources”: [dict]} (if ID is None)




**Example**



```
pipeline_details = client.repository.get_pipeline_details(pipeline_id)
pipeline_details = client.repository.get_pipeline_details()
pipeline_details = client.repository.get_pipeline_details(limit=100)
pipeline_details = client.repository.get_pipeline_details(limit=100, get_all=True)
pipeline_details = []
for entry in client.repository.get_pipeline_details(limit=100, asynchronous=True, get_all=True):
    pipeline_details.extend(entry)

```





*static* get\_pipeline\_href(*pipeline\_details*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_pipeline_href)[¶](#client.Repository.get_pipeline_href "Link to this definition")
Get the href from pipeline details.



Parameters:
**pipeline\_details** (*dict*) – metadata of the stored pipeline



Returns:
href of the pipeline



Return type:
str




**Example**



```
pipeline_details = client.repository.get_pipeline_details(pipeline_id)
pipeline_href = client.repository.get_pipeline_href(pipeline_details)

```





*static* get\_pipeline\_id(*pipeline\_details*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_pipeline_id)[¶](#client.Repository.get_pipeline_id "Link to this definition")
Get the pipeline ID from pipeline details.



Parameters:
**pipeline\_details** (*dict*) – metadata of the stored pipeline



Returns:
unique ID of the pipeline



Return type:
str




**Example**



```
pipeline_id = client.repository.get_pipeline_id(pipeline_details)

```





get\_pipeline\_revision\_details(*pipeline\_id=None*, *rev\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.get_pipeline_revision_details)[¶](#client.Repository.get_pipeline_revision_details "Link to this definition")
Get metadata of a pipeline revision.



Parameters:
* **pipeline\_id** (*str*) – ID of the stored pipeline
* **rev\_id** (*str*) – revision ID of the stored pipeline



Returns:
revised metadata of the stored pipeline



Return type:
dict




**Example:**



```
pipeline_details = client.repository.get_pipeline_revision_details(pipeline_id, rev_id)

```



Note


rev\_id parameter is not applicable in Cloud platform.






list(*framework\_filter=None*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.list)[¶](#client.Repository.list "Link to this definition")
Get and list stored models, pipelines, runtimes, libraries, functions, spaces, and experiments in a table/DataFrame format.
If limit is set to None, only the first 50 records are shown.



Parameters:
**framework\_filter** (*str**,* *optional*) – get only the frameworks with the desired names



Returns:
DataFrame with listed names and IDs of stored models



Return type:
pandas.DataFrame




**Example**



```
client.repository.list()
client.repository.list(framework_filter='prompt_tune')

```





list\_experiments(*limit=None*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.list_experiments)[¶](#client.Repository.list_experiments "Link to this definition")
List stored experiments in a table format.
If limit is set to None, only the first 50 records are shown.



Parameters:
**limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed experiments



Return type:
pandas.DataFrame




**Example**



```
client.repository.list_experiments()

```





list\_experiments\_revisions(*experiment\_id=None*, *limit=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.list_experiments_revisions)[¶](#client.Repository.list_experiments_revisions "Link to this definition")
Print all revisions for a given experiment ID in a table format.



Parameters:
* **experiment\_id** (*str*) – unique ID of the stored experiment
* **limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed revisions



Return type:
pandas.DataFrame




**Example**



```
client.repository.list_experiments_revisions(experiment_id)

```





list\_functions(*limit=None*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.list_functions)[¶](#client.Repository.list_functions "Link to this definition")
Return stored functions in a table format. If limit is set to None, only the first 50 records are shown.



Parameters:
**limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed functions



Return type:
pandas.DataFrame




**Example**



```
client.repository.list_functions()

```





list\_functions\_revisions(*function\_id=None*, *limit=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.list_functions_revisions)[¶](#client.Repository.list_functions_revisions "Link to this definition")
Print all revisions for a given function ID in a table format.



Parameters:
* **function\_id** (*str*) – unique ID of the stored function
* **limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed revisions



Return type:
pandas.DataFrame




**Example**



```
client.repository.list_functions_revisions(function_id)

```





list\_models(*limit=None*, *asynchronous=False*, *get\_all=False*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.list_models)[¶](#client.Repository.list_models "Link to this definition")
List stored models in a table format. If limit is set to None, only the first 50 records are shown.



Parameters:
* **limit** (*int**,* *optional*) – limit number of fetched records
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks



Returns:
pandas.DataFrame with listed models or generator if asynchronous is set to True



Return type:
pandas.DataFrame | Generator




**Example**



```
client.repository.list_models()
client.repository.list_models(limit=100)
client.repository.list_models(limit=100, get_all=True)
[entry for entry in client.repository.list_models(limit=100, asynchronous=True, get_all=True)]

```





list\_models\_revisions(*model\_id=None*, *limit=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.list_models_revisions)[¶](#client.Repository.list_models_revisions "Link to this definition")
Print all revisions for the given model ID in a table format.



Parameters:
* **model\_id** (*str*) – unique ID of the stored model
* **limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed revisions



Return type:
pandas.DataFrame




**Example**



```
client.repository.list_models_revisions(model_id)

```





list\_pipelines(*limit=None*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.list_pipelines)[¶](#client.Repository.list_pipelines "Link to this definition")
List stored pipelines in a table format. If limit is set to None, only the first 50 records are shown.



Parameters:
**limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed pipelines



Return type:
pandas.DataFrame




**Example**



```
client.repository.list_pipelines()

```





list\_pipelines\_revisions(*pipeline\_id=None*, *limit=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.list_pipelines_revisions)[¶](#client.Repository.list_pipelines_revisions "Link to this definition")
List all revision for a given pipeline ID in a table format.



Parameters:
* **pipeline\_id** (*str*) – unique ID of the stored pipeline
* **limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed revisions



Return type:
pandas.DataFrame




**Example**



```
client.repository.list_pipelines_revisions(pipeline_id)

```





load(*artifact\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.load)[¶](#client.Repository.load "Link to this definition")
Load a model from the repository to object in a local environment.



Parameters:
**artifact\_id** (*str*) – ID of the stored model



Returns:
trained model



Return type:
object




**Example**



```
model = client.models.load(model_id)

```





promote\_model(*model\_id*, *source\_project\_id*, *target\_space\_id*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.promote_model)[¶](#client.Repository.promote_model "Link to this definition")
Promote a model from a project to space. Supported only for IBM Cloud Pak® for Data.


*Deprecated:* Use client.spaces.promote(asset\_id, source\_project\_id, target\_space\_id) instead.





store\_experiment(*meta\_props*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.store_experiment)[¶](#client.Repository.store_experiment "Link to this definition")
Create an experiment.



Parameters:
**meta\_props** (*dict*) – metadata of the experiment configuration. To see available meta names, use:



```
client.repository.ExperimentMetaNames.get()

```






Returns:
metadata of the stored experiment



Return type:
dict




**Example**



```
metadata = {
    client.repository.ExperimentMetaNames.NAME: 'my_experiment',
    client.repository.ExperimentMetaNames.EVALUATION_METRICS: ['accuracy'],
    client.repository.ExperimentMetaNames.TRAINING_REFERENCES: [
        {'pipeline': {'href': pipeline_href_1}},
        {'pipeline': {'href':pipeline_href_2}}
    ]
}
experiment_details = client.repository.store_experiment(meta_props=metadata)
experiment_href = client.repository.get_experiment_href(experiment_details)

```





store\_function(*function*, *meta\_props*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.store_function)[¶](#client.Repository.store_function "Link to this definition")
Create a function.



As a ‘function’ may be used one of the following:* filepath to gz file
* ‘score’ function reference, where the function is the function which will be deployed
* generator function, which takes no argument or arguments which all have primitive python default values
and as result return ‘score’ function





Parameters:
* **function** (*str* *or* *function*) – path to file with archived function content or function (as described above)
* **meta\_props** (*str* *or* *dict*) – meta data or name of the function, to see available meta names
use `client.repository.FunctionMetaNames.show()`



Returns:
stored function metadata



Return type:
dict




**Examples**


The most simple use is (using score function):



```
meta_props = {
    client.repository.FunctionMetaNames.NAME: "function",
    client.repository.FunctionMetaNames.DESCRIPTION: "This is ai function",
    client.repository.FunctionMetaNames.SOFTWARE_SPEC_UID: "53dc4cf1-252f-424b-b52d-5cdd9814987f"}

def score(payload):
    values = [[row[0]*row[1]] for row in payload['values']]
    return {'fields': ['multiplication'], 'values': values}

stored_function_details = client.repository.store_function(score, meta_props)

```


Other, more interesting example is using generator function.
In this situation it is possible to pass some variables:



```
creds = {...}

def gen_function(credentials=creds, x=2):
    def f(payload):
        values = [[row[0]*row[1]*x] for row in payload['values']]
        return {'fields': ['multiplication'], 'values': values}
    return f

stored_function_details = client.repository.store_function(gen_function, meta_props)

```





store\_model(*model=None*, *meta\_props=None*, *training\_data=None*, *training\_target=None*, *pipeline=None*, *feature\_names=None*, *label\_column\_names=None*, *subtrainingId=None*, *round\_number=None*, *experiment\_metadata=None*, *training\_id=None*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.store_model)[¶](#client.Repository.store_model "Link to this definition")
Create a model.



Parameters:
* **model** (*str* *(**for filename* *or* *path**) or* *object* *(**corresponding to model type**)*) – Can be one of following:



	+ The train model object:
		- scikit-learn
		- xgboost
		- spark (PipelineModel)
	+ path to saved model in format:
	
	
	
	> - keras (.tgz)
	> 	- pmml (.xml)
	> 	- scikit-learn (.tar.gz)
	> 	- tensorflow (.tar.gz)
	> 	- spss (.str)
	> 	- spark (.tar.gz)
	+ directory containing model file(s):
	
	
	
	> - scikit-learn
	> 	- xgboost
	> 	- tensorflow
	+ unique ID of the trained model
* **meta\_props** (*dict**,* *optional*) – metadata of the models configuration. To see available meta names, use:



```
client.repository.ModelMetaNames.get()

```
* **training\_data** (*spark dataframe**,* *pandas dataframe**,* *numpy.ndarray* *or* *array**,* *optional*) – Spark DataFrame supported for spark models. Pandas dataframe, numpy.ndarray or array
supported for scikit-learn models
* **training\_target** (*array**,* *optional*) – array with labels required for scikit-learn models
* **pipeline** (*object**,* *optional*) – pipeline required for spark mllib models
* **feature\_names** (*numpy.ndarray* *or* *list**,* *optional*) – feature names for the training data in case of Scikit-Learn/XGBoost models,
this is applicable only in the case where the training data is not of type - pandas.DataFrame
* **label\_column\_names** (*numpy.ndarray* *or* *list**,* *optional*) – label column names of the trained Scikit-Learn/XGBoost models
* **round\_number** (*int**,* *optional*) – round number of a Federated Learning experiment that has been configured to save
intermediate models, this applies when model is a training id
* **experiment\_metadata** (*dict**,* *optional*) – metadata retrieved from the experiment that created the model
* **training\_id** (*str**,* *optional*) – Run id of AutoAI or TuneExperiment experiment.



Returns:
metadata of the created model



Return type:
dict





Note


* For a keras model, model content is expected to contain a .h5 file and an archived version of it.
* feature\_names is an optional argument containing the feature names for the training data
in case of Scikit-Learn/XGBoost models. Valid types are numpy.ndarray and list.
This is applicable only in the case where the training data is not of type - pandas.DataFrame.
* If the training\_data is of type pandas.DataFrame and feature\_names are provided,
feature\_names are ignored.
* For INPUT\_DATA\_SCHEMA meta prop use list even when passing single input data schema. You can provide
multiple schemas as dictionaries inside a list.



**Examples**



```
stored_model_details = client.repository.store_model(model, name)

```


In more complicated cases you should create proper metadata, similar to this one:



```
sw_spec_id = client.software_specifications.get_id_by_name('scikit-learn_0.23-py3.7')

metadata = {
    client.repository.ModelMetaNames.NAME: 'customer satisfaction prediction model',
    client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
    client.repository.ModelMetaNames.TYPE: 'scikit-learn_0.23'
}

```


In case when you want to provide input data schema of the model, you can provide it as part of meta:



```
sw_spec_id = client.software_specifications.get_id_by_name('spss-modeler_18.1')

metadata = {
    client.repository.ModelMetaNames.NAME: 'customer satisfaction prediction model',
    client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
    client.repository.ModelMetaNames.TYPE: 'spss-modeler_18.1',
    client.repository.ModelMetaNames.INPUT_DATA_SCHEMA: [{'id': 'test',
                                                          'type': 'list',
                                                          'fields': [{'name': 'age', 'type': 'float'},
                                                                     {'name': 'sex', 'type': 'float'},
                                                                     {'name': 'fbs', 'type': 'float'},
                                                                     {'name': 'restbp', 'type': 'float'}]
                                                          },
                                                          {'id': 'test2',
                                                           'type': 'list',
                                                           'fields': [{'name': 'age', 'type': 'float'},
                                                                      {'name': 'sex', 'type': 'float'},
                                                                      {'name': 'fbs', 'type': 'float'},
                                                                      {'name': 'restbp', 'type': 'float'}]
    }]
}

```


`store_model()` method used with a local tar.gz file that contains a model:



```
stored_model_details = client.repository.store_model(path_to_tar_gz, meta_props=metadata, training_data=None)

```


`store_model()` method used with a local directory that contains model files:



```
stored_model_details = client.repository.store_model(path_to_model_directory, meta_props=metadata, training_data=None)

```


`store_model()` method used with the ID of a trained model:



```
stored_model_details = client.repository.store_model(trained_model_id, meta_props=metadata, training_data=None)

```


`store_model()` method used with a pipeline that was generated by an AutoAI experiment:



```
metadata = {
    client.repository.ModelMetaNames.NAME: 'AutoAI prediction model stored from object'
}
stored_model_details = client.repository.store_model(pipeline_model, meta_props=metadata, experiment_metadata=experiment_metadata)

```



```
metadata = {
    client.repository.ModelMetaNames.NAME: 'AutoAI prediction Pipeline_1 model'
}
stored_model_details = client.repository.store_model(model="Pipeline_1", meta_props=metadata, training_id = training_id)

```


Example of storing a prompt tuned model:
.. code-block:: python



> stored\_model\_details = client.repository.store\_model(training\_id = prompt\_tuning\_run\_id)


Example of storing a custom foundation model:



```
sw_spec_id = client.software_specifications.get_id_by_name('watsonx-cfm-caikit-1.0')

metadata = {
    client.repository.ModelMetaNames.NAME: 'custom FM asset',
    client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
    client.repository.ModelMetaNames.TYPE: client.repository.ModelAssetTypes.CUSTOM_FOUNDATION_MODEL_1_0
}
stored_model_details = client.repository.store_model(model='mistralai/Mistral-7B-Instruct-v0.2', meta_props=metadata)

```





store\_pipeline(*meta\_props*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.store_pipeline)[¶](#client.Repository.store_pipeline "Link to this definition")
Create a pipeline.



Parameters:
**meta\_props** (*dict*) – metadata of the pipeline configuration. To see available meta names, use:



```
client.repository.PipelineMetaNames.get()

```






Returns:
stored pipeline metadata



Return type:
dict




**Example**



```
metadata = {
    client.repository.PipelineMetaNames.NAME: 'my_training_definition',
    client.repository.PipelineMetaNames.DOCUMENT: {"doc_type":"pipeline",
                                                       "version": "2.0",
                                                       "primary_pipeline": "dlaas_only",
                                                       "pipelines": [{"id": "dlaas_only",
                                                                      "runtime_ref": "hybrid",
                                                                      "nodes": [{"id": "training",
                                                                                 "type": "model_node",
                                                                                 "op": "dl_train",
                                                                                 "runtime_ref": "DL",
                                                                                 "inputs": [],
                                                                                 "outputs": [],
                                                                                 "parameters": {"name": "tf-mnist",
                                                                                                "description": "Simple MNIST model implemented in TF",
                                                                                                "command": "python3 convolutional_network.py --trainImagesFile ${DATA_DIR}/train-images-idx3-ubyte.gz --trainLabelsFile ${DATA_DIR}/train-labels-idx1-ubyte.gz --testImagesFile ${DATA_DIR}/t10k-images-idx3-ubyte.gz --testLabelsFile ${DATA_DIR}/t10k-labels-idx1-ubyte.gz --learningRate 0.001 --trainingIters 6000",
                                                                                                "compute": {"name": "k80","nodes": 1},
                                                                                                "training_lib_href": "/v4/libraries/64758251-bt01-4aa5-a7ay-72639e2ff4d2/content"
                                                                                 },
                                                                                 "target_bucket": "wml-dev-results"
                                                                      }]
                                                       }]
    }
}
pipeline_details = client.repository.store_pipeline(training_definition_filepath, meta_props=metadata)

```





update\_experiment(*experiment\_id=None*, *changes=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.update_experiment)[¶](#client.Repository.update_experiment "Link to this definition")
Updates existing experiment metadata.



Parameters:
* **experiment\_id** (*str*) – ID of the experiment with the definition to be updated
* **changes** (*dict*) – elements to be changed, where keys are ExperimentMetaNames



Returns:
metadata of the updated experiment



Return type:
dict




**Example**



```
metadata = {
    client.repository.ExperimentMetaNames.NAME: "updated_exp"
}
exp_details = client.repository.update_experiment(experiment_id, changes=metadata)

```





update\_function(*function\_id*, *changes=None*, *update\_function=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.update_function)[¶](#client.Repository.update_function "Link to this definition")
Updates existing function metadata.



Parameters:
* **function\_id** (*str*) – ID of function which define what should be updated
* **changes** (*dict*) – elements which should be changed, where keys are FunctionMetaNames
* **update\_function** (*str* *or* *function**,* *optional*) – path to file with archived function content or function which should be changed
for specific function\_id, this parameter is valid only for CP4D 3.0.0




**Example**



```
metadata = {
    client.repository.FunctionMetaNames.NAME: "updated_function"
}

function_details = client.repository.update_function(function_id, changes=metadata)

```





update\_model(*model\_id=None*, *updated\_meta\_props=None*, *update\_model=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.update_model)[¶](#client.Repository.update_model "Link to this definition")
Update an existing model.



Parameters:
* **model\_id** (*str*) – ID of model to be updated
* **updated\_meta\_props** (*dict**,* *optional*) – new set of updated\_meta\_props to be updated
* **update\_model** (*object* *or* *model**,* *optional*) – archived model content file or path to directory that contains the archived model file
that needs to be changed for the specific model\_id



Returns:
updated metadata of the model



Return type:
dict




**Example**



```
model_details = client.repository.update_model(model_id, update_model=updated_content)

```





update\_pipeline(*pipeline\_id=None*, *changes=None*, *rev\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/repository.html#Repository.update_pipeline)[¶](#client.Repository.update_pipeline "Link to this definition")
Update metadata of an existing pipeline.



Parameters:
* **pipeline\_id** (*str*) – unique ID of the pipeline to be updated
* **changes** (*dict*) – elements to be changed, where keys are PipelineMetaNames
* **rev\_id** (*str*) – revision ID of the pipeline



Returns:
metadata of the updated pipeline



Return type:
dict




**Example**



```
metadata = {
    client.repository.PipelineMetaNames.NAME: "updated_pipeline"
}
pipeline_details = client.repository.update_pipeline(pipeline_id, changes=metadata)

```






*class* metanames.ModelMetaNames[[source]](_modules/metanames.html#ModelMetaNames)[¶](#metanames.ModelMetaNames "Link to this definition")
Set of MetaNames for models.


Available MetaNames:





| MetaName | Type | Required | Schema | Example value |
| --- | --- | --- | --- | --- |
| NAME | str | Y |  | `my_model` |
| DESCRIPTION | str | N |  | `my_description` |
| INPUT\_DATA\_SCHEMA | list | N | `{'id(required)': 'string', 'fields(required)': [{'name(required)': 'string', 'type(required)': 'string', 'nullable(optional)': 'string'}]}` | `{'id': '1', 'type': 'struct', 'fields': [{'name': 'x', 'type': 'double', 'nullable': False, 'metadata': {}}, {'name': 'y', 'type': 'double', 'nullable': False, 'metadata': {}}]}` |
| TRAINING\_DATA\_REFERENCES | list | N | `[{'name(optional)': 'string', 'type(required)': 'string', 'connection(required)': {'endpoint_url(required)': 'string', 'access_key_id(required)': 'string', 'secret_access_key(required)': 'string'}, 'location(required)': {'bucket': 'string', 'path': 'string'}, 'schema(optional)': {'id(required)': 'string', 'fields(required)': [{'name(required)': 'string', 'type(required)': 'string', 'nullable(optional)': 'string'}]}}]` | `[]` |
| TEST\_DATA\_REFERENCES | list | N | `[{'name(optional)': 'string', 'type(required)': 'string', 'connection(required)': {'endpoint_url(required)': 'string', 'access_key_id(required)': 'string', 'secret_access_key(required)': 'string'}, 'location(required)': {'bucket': 'string', 'path': 'string'}, 'schema(optional)': {'id(required)': 'string', 'fields(required)': [{'name(required)': 'string', 'type(required)': 'string', 'nullable(optional)': 'string'}]}}]` | `[]` |
| OUTPUT\_DATA\_SCHEMA | dict | N | `{'id(required)': 'string', 'fields(required)': [{'name(required)': 'string', 'type(required)': 'string', 'nullable(optional)': 'string'}]}` | `{'id': '1', 'type': 'struct', 'fields': [{'name': 'x', 'type': 'double', 'nullable': False, 'metadata': {}}, {'name': 'y', 'type': 'double', 'nullable': False, 'metadata': {}}]}` |
| LABEL\_FIELD | str | N |  | `PRODUCT_LINE` |
| TRANSFORMED\_LABEL\_FIELD | str | N |  | `PRODUCT_LINE_IX` |
| TAGS | list | N | `['string', 'string']` | `['string', 'string']` |
| SIZE | dict | N | `{'in_memory(optional)': 'string', 'content(optional)': 'string'}` | `{'in_memory': 0, 'content': 0}` |
| PIPELINE\_ID | str | N |  | `53628d69-ced9-4f43-a8cd-9954344039a8` |
| RUNTIME\_ID | str | N |  | `53628d69-ced9-4f43-a8cd-9954344039a8` |
| TYPE | str | Y |  | `mllib_2.1` |
| CUSTOM | dict | N |  | `{}` |
| DOMAIN | str | N |  | `Watson Machine Learning` |
| HYPER\_PARAMETERS | dict | N |  |  |
| METRICS | list | N |  |  |
| IMPORT | dict | N | `{'name(optional)': 'string', 'type(required)': 'string', 'connection(required)': {'endpoint_url(required)': 'string', 'access_key_id(required)': 'string', 'secret_access_key(required)': 'string'}, 'location(required)': {'bucket': 'string', 'path': 'string'}}` | `{'connection': {'endpoint_url': 'https://s3-api.us-geo.objectstorage.softlayer.net', 'access_key_id': '***', 'secret_access_key': '***'}, 'location': {'bucket': 'train-data', 'path': 'training_path'}, 'type': 's3'}` |
| TRAINING\_LIB\_ID | str | N |  | `53628d69-ced9-4f43-a8cd-9954344039a8` |
| MODEL\_DEFINITION\_ID | str | N |  | `53628d6_cdee13-35d3-s8989343` |
| SOFTWARE\_SPEC\_ID | str | N |  | `53628d69-ced9-4f43-a8cd-9954344039a8` |
| TF\_MODEL\_PARAMS | dict | N |  | `{'save_format': 'None', 'signatures': 'struct', 'options': 'None', 'custom_objects': 'string'}` |
| FAIRNESS\_INFO | dict | N |  | `{'favorable_labels': ['X']}` |



**Note:** project (MetaNames.PROJECT\_ID) and space (MetaNames.SPACE\_ID) meta names are not supported and considered as invalid. Instead use client.set.default\_space(<SPACE\_ID>) to set the space or client.set.default\_project(<PROJECT\_ID>).





*class* metanames.ExperimentMetaNames[[source]](_modules/metanames.html#ExperimentMetaNames)[¶](#metanames.ExperimentMetaNames "Link to this definition")
Set of MetaNames for experiments.


Available MetaNames:





| MetaName | Type | Required | Schema | Example value |
| --- | --- | --- | --- | --- |
| NAME | str | Y |  | `Hand-written Digit Recognition` |
| DESCRIPTION | str | N |  | `Hand-written Digit Recognition training` |
| TAGS | list | N | `[{'value(required)': 'string', 'description(optional)': 'string'}]` | `[{'value': 'dsx-project.<project-guid>', 'description': 'DSX project guid'}]` |
| EVALUATION\_METHOD | str | N |  | `multiclass` |
| EVALUATION\_METRICS | list | N | `[{'name(required)': 'string', 'maximize(optional)': 'boolean'}]` | `[{'name': 'accuracy', 'maximize': False}]` |
| TRAINING\_REFERENCES | list | Y | `[{'pipeline(optional)': {'href(required)': 'string', 'data_bindings(optional)': [{'data_reference(required)': 'string', 'node_id(required)': 'string'}], 'nodes_parameters(optional)': [{'node_id(required)': 'string', 'parameters(required)': 'dict'}]}, 'training_lib(optional)': {'href(required)': 'string', 'compute(optional)': {'name(required)': 'string', 'nodes(optional)': 'number'}, 'runtime(optional)': {'href(required)': 'string'}, 'command(optional)': 'string', 'parameters(optional)': 'dict'}}]` | `[{'pipeline': {'href': '/v4/pipelines/6d758251-bb01-4aa5-a7a3-72339e2ff4d8'}}]` |
| SPACE\_UID | str | N |  | `3c1ce536-20dc-426e-aac7-7284cf3befc6` |
| LABEL\_COLUMN | str | N |  | `label` |
| CUSTOM | dict | N |  | `{'field1': 'value1'}` |






*class* metanames.FunctionMetaNames[[source]](_modules/metanames.html#FunctionMetaNames)[¶](#metanames.FunctionMetaNames "Link to this definition")
Set of MetaNames for AI functions.


Available MetaNames:





| MetaName | Type | Required | Schema | Example value |
| --- | --- | --- | --- | --- |
| NAME | str | Y |  | `ai_function` |
| DESCRIPTION | str | N |  | `This is ai function` |
| SOFTWARE\_SPEC\_ID | str | N |  | `53628d69-ced9-4f43-a8cd-9954344039a8` |
| SOFTWARE\_SPEC\_UID | str | N |  | `53628d69-ced9-4f43-a8cd-9954344039a8` |
| INPUT\_DATA\_SCHEMAS | list | N | `[{'id(required)': 'string', 'fields(required)': [{'name(required)': 'string', 'type(required)': 'string', 'nullable(optional)': 'string'}]}]` | `[{'id': '1', 'type': 'struct', 'fields': [{'name': 'x', 'type': 'double', 'nullable': False, 'metadata': {}}, {'name': 'y', 'type': 'double', 'nullable': False, 'metadata': {}}]}]` |
| OUTPUT\_DATA\_SCHEMAS | list | N | `[{'id(required)': 'string', 'fields(required)': [{'name(required)': 'string', 'type(required)': 'string', 'nullable(optional)': 'string'}]}]` | `[{'id': '1', 'type': 'struct', 'fields': [{'name': 'multiplication', 'type': 'double', 'nullable': False, 'metadata': {}}]}]` |
| TAGS | list | N | `['string']` | `['tags1', 'tags2']` |
| TYPE | str | N |  | `python` |
| CUSTOM | dict | N |  | `{}` |
| SAMPLE\_SCORING\_INPUT | dict | N | `{'id(optional)': 'string', 'fields(optional)': 'array', 'values(optional)': 'array'}` | `{'input_data': [{'fields': ['name', 'age', 'occupation'], 'values': [['john', 23, 'student'], ['paul', 33, 'engineer']]}]}` |






*class* metanames.PipelineMetanames[[source]](_modules/metanames.html#PipelineMetanames)[¶](#metanames.PipelineMetanames "Link to this definition")
Set of MetaNames for pipelines.


Available MetaNames:





| MetaName | Type | Required | Schema | Example value |
| --- | --- | --- | --- | --- |
| NAME | str | Y |  | `Hand-written Digit Recognitionu` |
| DESCRIPTION | str | N |  | `Hand-written Digit Recognition training` |
| SPACE\_UID | str | N |  | `3c1ce536-20dc-426e-aac7-7284cf3befc6` |
| TAGS | list | N | `[{'value(required)': 'string', 'description(optional)': 'string'}]` | `[{'value': 'dsx-project.<project-guid>', 'description': 'DSX project guid'}]` |
| DOCUMENT | dict | N | `{'doc_type(required)': 'string', 'version(required)': 'string', 'primary_pipeline(required)': 'string', 'pipelines(required)': [{'id(required)': 'string', 'runtime_ref(required)': 'string', 'nodes(required)': [{'id': 'string', 'type': 'string', 'inputs': 'list', 'outputs': 'list', 'parameters': {'training_lib_href': 'string'}}]}]}` | `{'doc_type': 'pipeline', 'version': '2.0', 'primary_pipeline': 'dlaas_only', 'pipelines': [{'id': 'dlaas_only', 'runtime_ref': 'hybrid', 'nodes': [{'id': 'training', 'type': 'model_node', 'op': 'dl_train', 'runtime_ref': 'DL', 'inputs': [], 'outputs': [], 'parameters': {'name': 'tf-mnist', 'description': 'Simple MNIST model implemented in TF', 'command': 'python3 convolutional_network.py --trainImagesFile ${DATA_DIR}/train-images-idx3-ubyte.gz --trainLabelsFile ${DATA_DIR}/train-labels-idx1-ubyte.gz --testImagesFile ${DATA_DIR}/t10k-images-idx3-ubyte.gz --testLabelsFile ${DATA_DIR}/t10k-labels-idx1-ubyte.gz --learningRate 0.001 --trainingIters 6000', 'compute': {'name': 'k80', 'nodes': 1}, 'training_lib_href': '/v4/libraries/64758251-bt01-4aa5-a7ay-72639e2ff4d2/content'}, 'target_bucket': 'wml-dev-results'}]}]}` |
| CUSTOM | dict | N |  | `{'field1': 'value1'}` |
| IMPORT | dict | N | `{'name(optional)': 'string', 'type(required)': 'string', 'connection(required)': {'endpoint_url(required)': 'string', 'access_key_id(required)': 'string', 'secret_access_key(required)': 'string'}, 'location(required)': {'bucket': 'string', 'path': 'string'}}` | `{'connection': {'endpoint_url': 'https://s3-api.us-geo.objectstorage.softlayer.net', 'access_key_id': '***', 'secret_access_key': '***'}, 'location': {'bucket': 'train-data', 'path': 'training_path'}, 'type': 's3'}` |
| RUNTIMES | list | N |  | `[{'id': 'id', 'name': 'tensorflow', 'version': '1.13-py3'}]` |
| COMMAND | str | N |  | `convolutional_network.py --trainImagesFile train-images-idx3-ubyte.gz --trainLabelsFile train-labels-idx1-ubyte.gz --testImagesFile t10k-images-idx3-ubyte.gz --testLabelsFile t10k-labels-idx1-ubyte.gz --learningRate 0.001 --trainingIters 6000` |
| LIBRARY\_UID | str | N |  | `fb9752c9-301a-415d-814f-cf658d7b856a` |
| COMPUTE | dict | N |  | `{'name': 'k80', 'nodes': 1}` |






Script[¶](#script "Link to this heading")
-----------------------------------------




*class* client.Script(*client*)[[source]](_modules/ibm_watsonx_ai/script.html#Script)[¶](#client.Script "Link to this definition")
Store and manage script assets.




ConfigurationMetaNames *= <ibm\_watsonx\_ai.metanames.ScriptMetaNames object>*[¶](#client.Script.ConfigurationMetaNames "Link to this definition")
MetaNames for script assets creation.





create\_revision(*script\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/script.html#Script.create_revision)[¶](#client.Script.create_revision "Link to this definition")
Create a revision for the given script. Revisions are immutable once created.
The metadata and attachment at script\_id is taken and a revision is created out of it.



Parameters:
**script\_id** (*str*) – ID of the script



Returns:
revised metadata of the stored script



Return type:
dict




**Example**



```
script_revision = client.script.create_revision(script_id)

```





delete(*asset\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/script.html#Script.delete)[¶](#client.Script.delete "Link to this definition")
Delete a stored script asset.



Parameters:
**asset\_id** (*str*) – ID of the script asset



Returns:
status (“SUCCESS” or “FAILED”) if deleted synchronously or dictionary with response



Return type:
str | dict




**Example**



```
client.script.delete(asset_id)

```





download(*asset\_id=None*, *filename=None*, *rev\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/script.html#Script.download)[¶](#client.Script.download "Link to this definition")
Download the content of a script asset.



Parameters:
* **asset\_id** (*str*) – unique ID of the script asset to be downloaded
* **filename** (*str*) – filename to be used for the downloaded file
* **rev\_id** (*str**,* *optional*) – revision ID



Returns:
path to the downloaded asset content



Return type:
str




**Example**



```
client.script.download(asset_id, "script_file")

```





get\_details(*script\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/script.html#Script.get_details)[¶](#client.Script.get_details "Link to this definition")
Get script asset details. If no script\_id is passed, details for all script assets are returned.



Parameters:
**script\_id** (*str**,* *optional*) – unique ID of the script



Returns:
metadata of the stored script asset



Return type:
* **dict** - if script\_id is not None
* **{“resources”: [dict]}** - if script\_id is None







**Example**



```
script_details = client.script.get_details(script_id)

```





*static* get\_href(*asset\_details*)[[source]](_modules/ibm_watsonx_ai/script.html#Script.get_href)[¶](#client.Script.get_href "Link to this definition")
Get the URL of a stored script asset.



Parameters:
**asset\_details** (*dict*) – details of the stored script asset



Returns:
href of the stored script asset



Return type:
str




**Example**



```
asset_details = client.script.get_details(asset_id)
asset_href = client.script.get_href(asset_details)

```





*static* get\_id(*asset\_details*)[[source]](_modules/ibm_watsonx_ai/script.html#Script.get_id)[¶](#client.Script.get_id "Link to this definition")
Get the unique ID of a stored script asset.



Parameters:
**asset\_details** (*dict*) – metadata of the stored script asset



Returns:
unique ID of the stored script asset



Return type:
str




**Example**



```
asset_id = client.script.get_id(asset_details)

```





get\_revision\_details(*script\_id=None*, *rev\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/script.html#Script.get_revision_details)[¶](#client.Script.get_revision_details "Link to this definition")
Get metadata of the script revision.



Parameters:
* **script\_id** (*str*) – ID of the script
* **rev\_id** (*str**,* *optional*) – ID of the revision. If this parameter is not provided, it returns the latest revision. If there is no latest revision, it returns an error.



Returns:
metadata of the stored script(s)



Return type:
list




**Example**



```
script_details = client.script.get_revision_details(script_id, rev_id)

```





list(*limit=None*)[[source]](_modules/ibm_watsonx_ai/script.html#Script.list)[¶](#client.Script.list "Link to this definition")
List stored scripts in a table format. If limit is set to None, only the first 50 records are shown.



Parameters:
**limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed scripts



Return type:
pandas.DataFrame




**Example**



```
client.script.list()

```





list\_revisions(*script\_id=None*, *limit=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/script.html#Script.list_revisions)[¶](#client.Script.list_revisions "Link to this definition")
Print all revisions for the given script ID in a table format.



Parameters:
* **script\_id** (*str*) – ID of the stored script
* **limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed revisions



Return type:
pandas.DataFrame




**Example**



```
client.script.list_revisions(script_id)

```





store(*meta\_props*, *file\_path*)[[source]](_modules/ibm_watsonx_ai/script.html#Script.store)[¶](#client.Script.store "Link to this definition")
Create a script asset and upload content to it.



Parameters:
* **meta\_props** (*dict*) – name to be given to the script asset
* **file\_path** (*str*) – path to the content file to be uploaded



Returns:
metadata of the stored script asset



Return type:
dict




**Example**



```
metadata = {
    client.script.ConfigurationMetaNames.NAME: 'my first script',
    client.script.ConfigurationMetaNames.DESCRIPTION: 'description of the script',
    client.script.ConfigurationMetaNames.SOFTWARE_SPEC_ID: '0cdb0f1e-5376-4f4d-92dd-da3b69aa9bda'
}

asset_details = client.script.store(meta_props=metadata, file_path="/path/to/file")

```





update(*script\_id=None*, *meta\_props=None*, *file\_path=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/script.html#Script.update)[¶](#client.Script.update "Link to this definition")
Update a script with metadata, attachment, or both.



Parameters:
* **script\_id** (*str*) – ID of the script
* **meta\_props** (*dict**,* *optional*) – changes for the script matadata
* **file\_path** (*str**,* *optional*) – file path to the new attachment



Returns:
updated metadata of the script



Return type:
dict




**Example**



```
script_details = client.script.update(script_id, meta, content_path)

```






*class* metanames.ScriptMetaNames[[source]](_modules/metanames.html#ScriptMetaNames)[¶](#metanames.ScriptMetaNames "Link to this definition")
Set of MetaNames for Script Specifications.


Available MetaNames:





| MetaName | Type | Required | Example value |
| --- | --- | --- | --- |
| NAME | str | Y | `Python script` |
| DESCRIPTION | str | N | `my_description` |
| SOFTWARE\_SPEC\_ID | str | Y | `53628d69-ced9-4f43-a8cd-9954344039a8` |






Service instance[¶](#service-instance "Link to this heading")
-------------------------------------------------------------




*class* client.ServiceInstance(*client*)[[source]](_modules/ibm_watsonx_ai/service_instance.html#ServiceInstance)[¶](#client.ServiceInstance "Link to this definition")
Connect, get details, and check usage of a Watson Machine Learning service instance.




get\_api\_key()[[source]](_modules/ibm_watsonx_ai/service_instance.html#ServiceInstance.get_api_key)[¶](#client.ServiceInstance.get_api_key "Link to this definition")
Get the API key of a Watson Machine Learning service.



Returns:
API key



Return type:
str




**Example**



```
instance_details = client.service_instance.get_api_key()

```





get\_details()[[source]](_modules/ibm_watsonx_ai/service_instance.html#ServiceInstance.get_details)[¶](#client.ServiceInstance.get_details "Link to this definition")
Get information about the Watson Machine Learning instance.



Returns:
metadata of the service instance



Return type:
dict




**Example**



```
instance_details = client.service_instance.get_details()

```





get\_instance\_id()[[source]](_modules/ibm_watsonx_ai/service_instance.html#ServiceInstance.get_instance_id)[¶](#client.ServiceInstance.get_instance_id "Link to this definition")
Get the instance ID of a Watson Machine Learning service.



Returns:
ID of the instance



Return type:
str




**Example**



```
instance_details = client.service_instance.get_instance_id()

```





get\_password()[[source]](_modules/ibm_watsonx_ai/service_instance.html#ServiceInstance.get_password)[¶](#client.ServiceInstance.get_password "Link to this definition")
Get the password for the Watson Machine Learning service. Applicable only for IBM Cloud Pak® for Data.



Returns:
password



Return type:
str




**Example**



```
instance_details = client.service_instance.get_password()

```





get\_url()[[source]](_modules/ibm_watsonx_ai/service_instance.html#ServiceInstance.get_url)[¶](#client.ServiceInstance.get_url "Link to this definition")
Get the instance URL of a Watson Machine Learning service.



Returns:
URL of the instance



Return type:
str




**Example**



```
instance_details = client.service_instance.get_url()

```





get\_username()[[source]](_modules/ibm_watsonx_ai/service_instance.html#ServiceInstance.get_username)[¶](#client.ServiceInstance.get_username "Link to this definition")
Get the username for the Watson Machine Learning service. Applicable only for IBM Cloud Pak® for Data.



Returns:
username



Return type:
str




**Example**



```
instance_details = client.service_instance.get_username()

```






Set[¶](#set "Link to this heading")
-----------------------------------




*class* client.Set(*client*)[[source]](_modules/ibm_watsonx_ai/Set.html#Set)[¶](#client.Set "Link to this definition")
Set a space\_id or a project\_id to be used in the subsequent actions.




default\_project(*project\_id*)[[source]](_modules/ibm_watsonx_ai/Set.html#Set.default_project)[¶](#client.Set.default_project "Link to this definition")
Set a project ID.



Parameters:
**project\_id** (*str*) – ID of the project to be used



Returns:
status (“SUCCESS” if succeeded)



Return type:
str




**Example**



```
client.set.default_project(project_id)

```





default\_space(*space\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/Set.html#Set.default_space)[¶](#client.Set.default_space "Link to this definition")
Set a space ID.



Parameters:
**space\_id** (*str*) – ID of the space to be used



Returns:
status (“SUCCESS” if succeeded)



Return type:
str




**Example**



```
client.set.default_space(space_id)

```






Shiny (IBM Cloud Pak for Data only)[¶](#shiny-ibm-cloud-pak-for-data-only "Link to this heading")
-------------------------------------------------------------------------------------------------


**Warning!** Not supported for IBM Cloud.




*class* client.Shiny(*client*)[[source]](_modules/ibm_watsonx_ai/shiny.html#Shiny)[¶](#client.Shiny "Link to this definition")
Store and manage shiny assets.




ConfigurationMetaNames *= <ibm\_watsonx\_ai.metanames.ShinyMetaNames object>*[¶](#client.Shiny.ConfigurationMetaNames "Link to this definition")
MetaNames for Shiny Assets creation.





create\_revision(*shiny\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/shiny.html#Shiny.create_revision)[¶](#client.Shiny.create_revision "Link to this definition")
Create a revision for the given shiny asset. Revisions are immutable once created.
The metadata and attachment at script\_id is taken and a revision is created out of it.



Parameters:
**shiny\_id** (*str*) – ID of the shiny asset



Returns:
revised metadata of the stored shiny asset



Return type:
dict




**Example**



```
shiny_revision = client.shiny.create_revision(shiny_id)

```





delete(*shiny\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/shiny.html#Shiny.delete)[¶](#client.Shiny.delete "Link to this definition")
Delete a stored shiny asset.



Parameters:
**shiny\_id** (*str*) – unique ID of the shiny asset



Returns:
status (“SUCCESS” or “FAILED”) if deleted synchronously or dictionary with response



Return type:
str | dict




**Example**



```
client.shiny.delete(shiny_id)

```





download(*shiny\_id=None*, *filename=None*, *rev\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/shiny.html#Shiny.download)[¶](#client.Shiny.download "Link to this definition")
Download the content of a shiny asset.



Parameters:
* **shiny\_id** (*str*) – unique ID of the shiny asset to be downloaded
* **filename** (*str*) – filename to be used for the downloaded file
* **rev\_id** (*str**,* *optional*) – ID of the revision



Returns:
path to the downloaded shiny asset content



Return type:
str




**Example**



```
client.shiny.download(shiny_id, "shiny_asset.zip")

```





get\_details(*shiny\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/shiny.html#Shiny.get_details)[¶](#client.Shiny.get_details "Link to this definition")
Get shiny asset details. If no shiny\_id is passed, details for all shiny assets are returned.



Parameters:
**shiny\_id** (*str**,* *optional*) – unique ID of the shiny asset



Returns:
metadata of the stored shiny asset



Return type:
* **dict** - if shiny\_id is not None
* **{“resources”: [dict]}** - if shiny\_id is None







**Example**



```
shiny_details = client.shiny.get_details(shiny_id)

```





*static* get\_href(*shiny\_details*)[[source]](_modules/ibm_watsonx_ai/shiny.html#Shiny.get_href)[¶](#client.Shiny.get_href "Link to this definition")
Get the URL of a stored shiny asset.



Parameters:
**shiny\_details** (*dict*) – details of the stored shiny asset



Returns:
href of the stored shiny asset



Return type:
str




**Example**



```
shiny_details = client.shiny.get_details(shiny_id)
shiny_href = client.shiny.get_href(shiny_details)

```





*static* get\_id(*shiny\_details*)[[source]](_modules/ibm_watsonx_ai/shiny.html#Shiny.get_id)[¶](#client.Shiny.get_id "Link to this definition")
Get the unique ID of a stored shiny asset.



Parameters:
**shiny\_details** (*dict*) – metadata of the stored shiny asset



Returns:
unique ID of the stored shiny asset



Return type:
str




**Example**



```
shiny_id = client.shiny.get_id(shiny_details)

```





get\_revision\_details(*shiny\_id=None*, *rev\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/shiny.html#Shiny.get_revision_details)[¶](#client.Shiny.get_revision_details "Link to this definition")
Get metadata of the shiny\_id revision.



Parameters:
* **shiny\_id** (*str*) – ID of the shiny asset
* **rev\_id** (*str**,* *optional*) – ID of the revision. If this parameter is not provided, it returns the latest revision. If there is no latest revision, it returns an error.



Returns:
stored shiny(s) metadata



Return type:
list




**Example**



```
shiny_details = client.shiny.get_revision_details(shiny_id, rev_id)

```





*static* get\_uid(*shiny\_details*)[[source]](_modules/ibm_watsonx_ai/shiny.html#Shiny.get_uid)[¶](#client.Shiny.get_uid "Link to this definition")
Get the Unique ID of a stored shiny asset.


*Deprecated:* Use `get_id(shiny_details)` instead.



Parameters:
**shiny\_details** (*dict*) – metadata of the stored shiny asset



Returns:
unique ID of the stored shiny asset



Return type:
str




**Example**



```
shiny_id = client.shiny.get_uid(shiny_details)

```





list(*limit=None*)[[source]](_modules/ibm_watsonx_ai/shiny.html#Shiny.list)[¶](#client.Shiny.list "Link to this definition")
List stored shiny assets in a table format. If limit is set to None,
only the first 50 records are shown.



Parameters:
**limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed shiny assets



Return type:
pandas.DataFrame




**Example**



```
client.shiny.list()

```





list\_revisions(*shiny\_id=None*, *limit=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/shiny.html#Shiny.list_revisions)[¶](#client.Shiny.list_revisions "Link to this definition")
List all revisions for the given shiny asset ID in a table format.



Parameters:
* **shiny\_id** (*str*) – ID of the stored shiny asset
* **limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed shiny revisions



Return type:
pandas.DataFrame




**Example**



```
client.shiny.list_revisions(shiny_id)

```





store(*meta\_props*, *file\_path*)[[source]](_modules/ibm_watsonx_ai/shiny.html#Shiny.store)[¶](#client.Shiny.store "Link to this definition")
Create a shiny asset and upload content to it.



Parameters:
* **meta\_props** (*dict*) – metadata of the shiny asset
* **file\_path** (*str*) – path to the content file to be uploaded



Returns:
metadata of the stored shiny asset



Return type:
dict




**Example**



```
meta_props = {
    client.shiny.ConfigurationMetaNames.NAME: "shiny app name"
}

shiny_details = client.shiny.store(meta_props, file_path="/path/to/file")

```





update(*shiny\_id=None*, *meta\_props=None*, *file\_path=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/shiny.html#Shiny.update)[¶](#client.Shiny.update "Link to this definition")
Update a shiny asset with metadata, attachment, or both.



Parameters:
* **shiny\_id** (*str*) – ID of the shiny asset
* **meta\_props** (*dict**,* *optional*) – changes to the metadata of the shiny asset
* **file\_path** (*str**,* *optional*) – file path to the new attachment



Returns:
updated metadata of the shiny asset



Return type:
dict




**Example**



```
script_details = client.script.update(shiny_id, meta, content_path)

```






Software specifications[¶](#software-specifications "Link to this heading")
---------------------------------------------------------------------------




*class* client.SwSpec(*client*)[[source]](_modules/ibm_watsonx_ai/sw_spec.html#SwSpec)[¶](#client.SwSpec "Link to this definition")
Store and manage software specs.




ConfigurationMetaNames *= <ibm\_watsonx\_ai.metanames.SwSpecMetaNames object>*[¶](#client.SwSpec.ConfigurationMetaNames "Link to this definition")
MetaNames for Software Specification creation.





add\_package\_extension(*sw\_spec\_id=None*, *pkg\_extn\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/sw_spec.html#SwSpec.add_package_extension)[¶](#client.SwSpec.add_package_extension "Link to this definition")
Add a package extension to a software specification’s existing metadata.



Parameters:
* **sw\_spec\_id** (*str*) – unique ID of the software specification to be updated
* **pkg\_extn\_id** (*str*) – unique ID of the package extension to be added to the software specification



Returns:
status



Return type:
str




**Example**



```
client.software_specifications.add_package_extension(sw_spec_id, pkg_extn_id)

```





delete(*sw\_spec\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/sw_spec.html#SwSpec.delete)[¶](#client.SwSpec.delete "Link to this definition")
Delete a software specification.



Parameters:
**sw\_spec\_id** (*str*) – unique ID of the software specification



Returns:
status (“SUCCESS” or “FAILED”)



Return type:
str




**Example**



```
client.software_specifications.delete(sw_spec_id)

```





delete\_package\_extension(*sw\_spec\_id=None*, *pkg\_extn\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/sw_spec.html#SwSpec.delete_package_extension)[¶](#client.SwSpec.delete_package_extension "Link to this definition")
Delete a package extension from a software specification’s existing metadata.



Parameters:
* **sw\_spec\_id** (*str*) – unique ID of the software specification to be updated
* **pkg\_extn\_id** (*str*) – unique ID of the package extension to be deleted from the software specification



Returns:
status



Return type:
str




**Example**



```
client.software_specifications.delete_package_extension(sw_spec_uid, pkg_extn_id)

```





get\_details(*sw\_spec\_id=None*, *state\_info=False*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/sw_spec.html#SwSpec.get_details)[¶](#client.SwSpec.get_details "Link to this definition")
Get software specification details. If no sw\_spec\_id is passed, details for all software specifications
are returned.



Parameters:
* **sw\_spec\_id** (*bool*) – ID of the software specification
* **state\_info** – works only when sw\_spec\_id is None, instead of returning details of software specs, it returns
the state of the software specs information (supported, unsupported, deprecated), containing suggested replacement
in case of unsupported or deprecated software specs



Returns:
metadata of the stored software specification(s)



Return type:
* **dict** - if sw\_spec\_uid is not None
* **{“resources”: [dict]}** - if sw\_spec\_uid is None







**Examples**



```
sw_spec_details = client.software_specifications.get_details(sw_spec_uid)
sw_spec_details = client.software_specifications.get_details()
sw_spec_state_details = client.software_specifications.get_details(state_info=True)

```





*static* get\_href(*sw\_spec\_details*)[[source]](_modules/ibm_watsonx_ai/sw_spec.html#SwSpec.get_href)[¶](#client.SwSpec.get_href "Link to this definition")
Get the URL of a software specification.



Parameters:
**sw\_spec\_details** (*dict*) – details of the software specification



Returns:
href of the software specification



Return type:
str




**Example**



```
sw_spec_details = client.software_specifications.get_details(sw_spec_id)
sw_spec_href = client.software_specifications.get_href(sw_spec_details)

```





*static* get\_id(*sw\_spec\_details*)[[source]](_modules/ibm_watsonx_ai/sw_spec.html#SwSpec.get_id)[¶](#client.SwSpec.get_id "Link to this definition")
Get the unique ID of a software specification.



Parameters:
**sw\_spec\_details** (*dict*) – metadata of the software specification



Returns:
unique ID of the software specification



Return type:
str




**Example**



```
asset_id = client.software_specifications.get_id(sw_spec_details)

```





get\_id\_by\_name(*sw\_spec\_name*)[[source]](_modules/ibm_watsonx_ai/sw_spec.html#SwSpec.get_id_by_name)[¶](#client.SwSpec.get_id_by_name "Link to this definition")
Get the unique ID of a software specification.



Parameters:
**sw\_spec\_name** (*str*) – name of the software specification



Returns:
unique ID of the software specification



Return type:
str




**Example**



```
asset_uid = client.software_specifications.get_id_by_name(sw_spec_name)

```





*static* get\_uid(*sw\_spec\_details*)[[source]](_modules/ibm_watsonx_ai/sw_spec.html#SwSpec.get_uid)[¶](#client.SwSpec.get_uid "Link to this definition")
Get the unique ID of a software specification.


*Deprecated:* Use `get_id(sw_spec_details)` instead.



Parameters:
**sw\_spec\_details** (*dict*) – metadata of the software specification



Returns:
unique ID of the software specification



Return type:
str




**Example**



```
asset_uid = client.software_specifications.get_uid(sw_spec_details)

```





get\_uid\_by\_name(*sw\_spec\_name*)[[source]](_modules/ibm_watsonx_ai/sw_spec.html#SwSpec.get_uid_by_name)[¶](#client.SwSpec.get_uid_by_name "Link to this definition")
Get the unique ID of a software specification.


*Deprecated:* Use `get_id_by_name(self, sw_spec_name)` instead.



Parameters:
**sw\_spec\_name** (*str*) – name of the software specification



Returns:
unique ID of the software specification



Return type:
str




**Example**



```
asset_uid = client.software_specifications.get_uid_by_name(sw_spec_name)

```





list(*limit=None*)[[source]](_modules/ibm_watsonx_ai/sw_spec.html#SwSpec.list)[¶](#client.SwSpec.list "Link to this definition")
List software specifications in a table format.



Parameters:
**limit** (*int**,* *optional*) – limit number of fetched records



Returns:
pandas.DataFrame with listed software specifications



Return type:
pandas.DataFrame




**Example**



```
client.software_specifications.list()

```





store(*meta\_props*)[[source]](_modules/ibm_watsonx_ai/sw_spec.html#SwSpec.store)[¶](#client.SwSpec.store "Link to this definition")
Create a software specification.



Parameters:
**meta\_props** (*dict*) – metadata of the space configuration. To see available meta names, use:



```
client.software_specifications.ConfigurationMetaNames.get()

```






Returns:
metadata of the stored space



Return type:
dict




**Example**



```
meta_props = {
    client.software_specifications.ConfigurationMetaNames.NAME: "skl_pipeline_heart_problem_prediction",
    client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "description scikit-learn_0.20",
    client.software_specifications.ConfigurationMetaNames.PACKAGE_EXTENSIONS_UID: [],
    client.software_specifications.ConfigurationMetaNames.SOFTWARE_CONFIGURATIONS: {},
    client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION_ID: "guid"
}

sw_spec_details = client.software_specifications.store(meta_props)

```






*class* metanames.SwSpecMetaNames[[source]](_modules/metanames.html#SwSpecMetaNames)[¶](#metanames.SwSpecMetaNames "Link to this definition")
Set of MetaNames for Software Specifications Specs.


Available MetaNames:





| MetaName | Type | Required | Schema | Example value |
| --- | --- | --- | --- | --- |
| NAME | str | Y |  | `Python 3.10 with pre-installed ML package` |
| DESCRIPTION | str | N |  | `my_description` |
| PACKAGE\_EXTENSIONS | list | N |  | `[{'guid': 'value'}]` |
| SOFTWARE\_CONFIGURATION | dict | N | `{'platform(required)': 'string'}` | `{'platform': {'name': 'python', 'version': '3.10'}}` |
| BASE\_SOFTWARE\_SPECIFICATION | dict | Y |  | `{'guid': 'BASE_SOFTWARE_SPECIFICATION_ID'}` |






Spaces[¶](#spaces "Link to this heading")
-----------------------------------------




*class* client.Spaces(*client*)[[source]](_modules/ibm_watsonx_ai/spaces.html#Spaces)[¶](#client.Spaces "Link to this definition")
Store and manage spaces.




ConfigurationMetaNames *= <ibm\_watsonx\_ai.metanames.SpacesMetaNames object>*[¶](#client.Spaces.ConfigurationMetaNames "Link to this definition")
MetaNames for spaces creation.





MemberMetaNames *= <ibm\_watsonx\_ai.metanames.SpacesMemberMetaNames object>*[¶](#client.Spaces.MemberMetaNames "Link to this definition")
MetaNames for space members creation.





create\_member(*space\_id*, *meta\_props*)[[source]](_modules/ibm_watsonx_ai/spaces.html#Spaces.create_member)[¶](#client.Spaces.create_member "Link to this definition")
Create a member within a space.



Parameters:
* **space\_id** (*str*) – ID of the space with the definition to be updated
* **meta\_props** (*dict*) – metadata of the member configuration. To see available meta names, use:



```
client.spaces.MemberMetaNames.get()

```



Returns:
metadata of the stored member



Return type:
dict





Note


* role can be any one of the following: “viewer”, “editor”, “admin”
* type can be any one of the following: “user”, “service”
* id can be one of the following: service-ID or IAM-userID



**Examples**



```
metadata = {
    client.spaces.MemberMetaNames.MEMBERS: [{"id":"IBMid-100000DK0B",
                                             "type": "user",
                                             "role": "admin" }]
}
members_details = client.spaces.create_member(space_id=space_id, meta_props=metadata)

```



```
metadata = {
    client.spaces.MemberMetaNames.MEMBERS: [{"id":"iam-ServiceId-5a216e59-6592-43b9-8669-625d341aca71",
                                             "type": "service",
                                             "role": "admin" }]
}
members_details = client.spaces.create_member(space_id=space_id, meta_props=metadata)

```





delete(*space\_id*)[[source]](_modules/ibm_watsonx_ai/spaces.html#Spaces.delete)[¶](#client.Spaces.delete "Link to this definition")
Delete a stored space.



Parameters:
**space\_id** (*str*) – ID of the space



Returns:
status “SUCCESS” if deletion is successful



Return type:
Literal[“SUCCESS”]




**Example**



```
client.spaces.delete(space_id)

```





delete\_member(*space\_id*, *member\_id*)[[source]](_modules/ibm_watsonx_ai/spaces.html#Spaces.delete_member)[¶](#client.Spaces.delete_member "Link to this definition")
Delete a member associated with a space.



Parameters:
* **space\_id** (*str*) – ID of the space
* **member\_id** (*str*) – ID of the member



Returns:
status (“SUCCESS” or “FAILED”)



Return type:
str




**Example**



```
client.spaces.delete_member(space_id,member_id)

```





get\_details(*space\_id=None*, *limit=None*, *asynchronous=False*, *get\_all=False*)[[source]](_modules/ibm_watsonx_ai/spaces.html#Spaces.get_details)[¶](#client.Spaces.get_details "Link to this definition")
Get metadata of stored space(s).



Parameters:
* **space\_id** (*str**,* *optional*) – ID of the space
* **limit** (*int**,* *optional*) – applicable when space\_id is not provided, otherwise limit will be ignored
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks



Returns:
metadata of stored space(s)



Return type:
dict




**Example**



```
space_details = client.spaces.get_details(space_id)
space_details = client.spaces.get_details(limit=100)
space_details = client.spaces.get_details(limit=100, get_all=True)
space_details = []
for entry in client.spaces.get_details(limit=100, asynchronous=True, get_all=True):
    space_details.extend(entry)

```





*static* get\_id(*space\_details*)[[source]](_modules/ibm_watsonx_ai/spaces.html#Spaces.get_id)[¶](#client.Spaces.get_id "Link to this definition")
Get the space\_id from the space details.



Parameters:
**space\_details** (*dict*) – metadata of the stored space



Returns:
ID of the stored space



Return type:
str




**Example**



```
space_details = client.spaces.store(meta_props)
space_id = client.spaces.get_id(space_details)

```





get\_member\_details(*space\_id*, *member\_id*)[[source]](_modules/ibm_watsonx_ai/spaces.html#Spaces.get_member_details)[¶](#client.Spaces.get_member_details "Link to this definition")
Get metadata of a member associated with a space.



Parameters:
* **space\_id** (*str*) – ID of that space with the definition to be updated
* **member\_id** (*str*) – ID of the member



Returns:
metadata of the space member



Return type:
dict




**Example**



```
member_details = client.spaces.get_member_details(space_id,member_id)

```





*static* get\_uid(*space\_details*)[[source]](_modules/ibm_watsonx_ai/spaces.html#Spaces.get_uid)[¶](#client.Spaces.get_uid "Link to this definition")
Get the unique ID of the space.



> *Deprecated:* Use `get_id(space_details)` instead.
> 
> 
> 
> param space\_details:
> metadata of the space
> 
> 
> 
> type space\_details:
> dict
> 
> 
> 
> return:
> unique ID of the space
> 
> 
> 
> rtype:
> str


**Example**



```
space_details = client.spaces.store(meta_props)
space_uid = client.spaces.get_uid(space_details)

```





list(*limit=None*, *member=None*, *roles=None*, *space\_type=None*)[[source]](_modules/ibm_watsonx_ai/spaces.html#Spaces.list)[¶](#client.Spaces.list "Link to this definition")
List stored spaces in a table format. If limit is set to None, only the first 50 records are shown.



Parameters:
* **limit** (*int**,* *optional*) – limit number of fetched records
* **member** (*str**,* *optional*) – filters the result list, only includes spaces where the user with a matching user ID
is a member
* **roles** (*str**,* *optional*) – limit number of fetched records
* **space\_type** (*str**,* *optional*) – filter spaces by their type, available types are ‘wx’, ‘cpd’, and ‘wca’



Returns:
pandas.DataFrame with listed spaces



Return type:
pandas.DataFrame




**Example**



```
client.spaces.list()

```





list\_members(*space\_id*, *limit=None*, *identity\_type=None*, *role=None*, *state=None*)[[source]](_modules/ibm_watsonx_ai/spaces.html#Spaces.list_members)[¶](#client.Spaces.list_members "Link to this definition")
Print the stored members of a space in a table format.
If limit is set to None, only the first 50 records are shown.



Parameters:
* **space\_id** (*str*) – ID of the space
* **limit** (*int**,* *optional*) – limit number of fetched records
* **identity\_type** (*str**,* *optional*) – filter the members by type
* **role** (*str**,* *optional*) – filter the members by role
* **state** (*str**,* *optional*) – filter the members by state



Returns:
pandas.DataFrame with listed members



Return type:
pandas.DataFrame




**Example**



```
client.spaces.list_members(space_id)

```





promote(*asset\_id*, *source\_project\_id*, *target\_space\_id*, *rev\_id=None*)[[source]](_modules/ibm_watsonx_ai/spaces.html#Spaces.promote)[¶](#client.Spaces.promote "Link to this definition")
Promote an asset from a project to a space.



Parameters:
* **asset\_id** (*str*) – ID of the stored asset
* **source\_project\_id** (*str*) – source project, from which the asset is promoted
* **target\_space\_id** (*str*) – target space, where the asset is promoted
* **rev\_id** (*str**,* *optional*) – revision ID of the promoted asset



Returns:
ID of the promoted asset



Return type:
str




**Examples**



```
promoted_asset_id = client.spaces.promote(asset_id, source_project_id=project_id, target_space_id=space_id)
promoted_model_id = client.spaces.promote(model_id, source_project_id=project_id, target_space_id=space_id)
promoted_function_id = client.spaces.promote(function_id, source_project_id=project_id, target_space_id=space_id)
promoted_data_asset_id = client.spaces.promote(data_asset_id, source_project_id=project_id, target_space_id=space_id)
promoted_connection_asset_id = client.spaces.promote(connection_id, source_project_id=project_id, target_space_id=space_id)

```





store(*meta\_props*, *background\_mode=True*)[[source]](_modules/ibm_watsonx_ai/spaces.html#Spaces.store)[¶](#client.Spaces.store "Link to this definition")
Create a space. The instance associated with the space via COMPUTE will be used for billing purposes on
the cloud. Note that STORAGE and COMPUTE are applicable only for cloud.



Parameters:
* **meta\_props** (*dict*) – metadata of the space configuration. To see available meta names, use:



```
client.spaces.ConfigurationMetaNames.get()

```
* **background\_mode** (*bool**,* *optional*) – indicator if store() method will run in background (async) or (sync)



Returns:
metadata of the stored space



Return type:
dict




**Example**



```
metadata = {
    client.spaces.ConfigurationMetaNames.NAME: "my_space",
    client.spaces.ConfigurationMetaNames.DESCRIPTION: "spaces",
    client.spaces.ConfigurationMetaNames.STORAGE: {"resource_crn": "provide crn of the COS storage"},
    client.spaces.ConfigurationMetaNames.COMPUTE: {"name": "test_instance",
                                                   "crn": "provide crn of the instance"},
    client.spaces.ConfigurationMetaNames.STAGE: {"production": True,
                                                 "name": "stage_name"},
    client.spaces.ConfigurationMetaNames.TAGS: ["sample_tag_1", "sample_tag_2"],
    client.spaces.ConfigurationMetaNames.TYPE: "cpd",
}
spaces_details = client.spaces.store(meta_props=metadata)

```





update(*space\_id*, *changes*)[[source]](_modules/ibm_watsonx_ai/spaces.html#Spaces.update)[¶](#client.Spaces.update "Link to this definition")
Update existing space metadata. ‘STORAGE’ cannot be updated.
STORAGE and COMPUTE are applicable only for cloud.



Parameters:
* **space\_id** (*str*) – ID of the space with the definition to be updated
* **changes** (*dict*) – elements to be changed, where keys are ConfigurationMetaNames



Returns:
metadata of the updated space



Return type:
dict




**Example**



```
metadata = {
    client.spaces.ConfigurationMetaNames.NAME:"updated_space",
    client.spaces.ConfigurationMetaNames.COMPUTE: {"name": "test_instance",
                                                   "crn": "v1:staging:public:pm-20-dev:us-south:a/09796a1b4cddfcc9f7fe17824a68a0f8:f1026e4b-77cf-4703-843d-c9984eac7272::"
    }
}
space_details = client.spaces.update(space_id, changes=metadata)

```





update\_member(*space\_id*, *member\_id*, *changes*)[[source]](_modules/ibm_watsonx_ai/spaces.html#Spaces.update_member)[¶](#client.Spaces.update_member "Link to this definition")
Update the metadata of an existing member.



Parameters:
* **space\_id** (*str*) – ID of the space
* **member\_id** (*str*) – ID of the member to be updated
* **changes** (*dict*) – elements to be changed, where keys are ConfigurationMetaNames



Returns:
metadata of the updated member



Return type:
dict




**Example**



```
metadata = {
    client.spaces.MemberMetaNames.MEMBER: {"role": "editor"}
}
member_details = client.spaces.update_member(space_id, member_id, changes=metadata)

```






*class* metanames.SpacesMetaNames[[source]](_modules/metanames.html#SpacesMetaNames)[¶](#metanames.SpacesMetaNames "Link to this definition")
Set of MetaNames for Platform Spaces Specs.


Available MetaNames:





| MetaName | Type | Required | Example value |
| --- | --- | --- | --- |
| NAME | str | Y | `my_space` |
| DESCRIPTION | str | N | `my_description` |
| STORAGE | dict | N | `{'type': 'bmcos_object_storage', 'resource_crn': '', 'delegated(optional)': 'false'}` |
| COMPUTE | dict | N | `{'name': 'name', 'crn': 'crn of the instance'}` |
| STAGE | dict | N | `{'production': True, 'name': 'name of the stage'}` |
| TAGS | list | N | `['sample_tag']` |
| TYPE | str | N | `cpd` |






*class* metanames.SpacesMemberMetaNames[[source]](_modules/metanames.html#SpacesMemberMetaNames)[¶](#metanames.SpacesMemberMetaNames "Link to this definition")
Set of MetaNames for Platform Spaces Member Specs.


Available MetaNames:





| MetaName | Type | Required | Schema | Example value |
| --- | --- | --- | --- | --- |
| MEMBERS | list | N | `[{'id(required)': 'string', 'role(required)': 'string', 'type(required)': 'string', 'state(optional)': 'string'}]` | `[{'id': 'iam-id1', 'role': 'editor', 'type': 'user', 'state': 'active'}, {'id': 'iam-id2', 'role': 'viewer', 'type': 'user', 'state': 'active'}]` |
| MEMBER | dict | N |  | `{'id': 'iam-id1', 'role': 'editor', 'type': 'user', 'state': 'active'}` |






Training[¶](#training "Link to this heading")
---------------------------------------------




*class* client.Training(*client*)[[source]](_modules/ibm_watsonx_ai/training.html#Training)[¶](#client.Training "Link to this definition")
Train new models.




cancel(*training\_id=None*, *hard\_delete=False*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/training.html#Training.cancel)[¶](#client.Training.cancel "Link to this definition")
Cancel a training that is currently running. This method can delete metadata
details of a completed or canceled training run when hard\_delete parameter is set to True.



Parameters:
* **training\_id** (*str*) – ID of the training
* **hard\_delete** (*bool**,* *optional*) – specify True or False:



	+ True - to delete the completed or canceled training run
	+ False - to cancel the currently running training run



Returns:
status “SUCCESS” if cancelation is successful



Return type:
Literal[“SUCCESS”]




**Example**



```
client.training.cancel(training_id)

```





get\_details(*training\_id=None*, *limit=None*, *asynchronous=False*, *get\_all=False*, *training\_type=None*, *state=None*, *tag\_value=None*, *training\_definition\_id=None*, *\_internal=False*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/training.html#Training.get_details)[¶](#client.Training.get_details "Link to this definition")
Get metadata of training(s). If training\_id is not specified, the metadata of all model spaces are returned.



Parameters:
* **training\_id** (*str**,* *optional*) – unique ID of the training
* **limit** (*int**,* *optional*) – limit number of fetched records
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks
* **training\_type** (*str**,* *optional*) – filter the fetched list of trainings based on the training type [“pipeline” or “experiment”]
* **state** (*str**,* *optional*) – filter the fetched list of training based on their state:
[queued, running, completed, failed]
* **tag\_value** (*str**,* *optional*) – filter the fetched list of training based on their tag value
* **training\_definition\_id** (*str**,* *optional*) – filter the fetched trainings that are using the given training definition



Returns:
metadata of training(s)



Return type:
* **dict** - if training\_id is not None
* **{“resources”: [dict]}** - if training\_id is None







**Examples**



```
training_run_details = client.training.get_details(training_id)
training_runs_details = client.training.get_details()
training_runs_details = client.training.get_details(limit=100)
training_runs_details = client.training.get_details(limit=100, get_all=True)
training_runs_details = []
for entry in client.training.get_details(limit=100, asynchronous=True, get_all=True):
    training_runs_details.extend(entry)

```





*static* get\_href(*training\_details*)[[source]](_modules/ibm_watsonx_ai/training.html#Training.get_href)[¶](#client.Training.get_href "Link to this definition")
Get the training href from the training details.



Parameters:
**training\_details** (*dict*) – metadata of the created training



Returns:
training href



Return type:
str




**Example**



```
training_details = client.training.get_details(training_id)
run_url = client.training.get_href(training_details)

```





*static* get\_id(*training\_details*)[[source]](_modules/ibm_watsonx_ai/training.html#Training.get_id)[¶](#client.Training.get_id "Link to this definition")
Get the training ID from the training details.



Parameters:
**training\_details** (*dict*) – metadata of the created training



Returns:
unique ID of the training



Return type:
str




**Example**



```
training_details = client.training.get_details(training_id)
training_id = client.training.get_id(training_details)

```





get\_metrics(*training\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/training.html#Training.get_metrics)[¶](#client.Training.get_metrics "Link to this definition")
Get metrics of a training run.



Parameters:
**training\_id** (*str*) – ID of the training



Returns:
metrics of the training run



Return type:
list of dict




**Example**



```
training_status = client.training.get_metrics(training_id)

```





get\_status(*training\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/training.html#Training.get_status)[¶](#client.Training.get_status "Link to this definition")
Get the status of a created training.



Parameters:
**training\_id** (*str*) – ID of the training



Returns:
training\_status



Return type:
dict




**Example**



```
training_status = client.training.get_status(training_id)

```





list(*limit=None*, *asynchronous=False*, *get\_all=False*)[[source]](_modules/ibm_watsonx_ai/training.html#Training.list)[¶](#client.Training.list "Link to this definition")
List stored trainings in a table format. If limit is set to None, only the first 50 records are shown.



Parameters:
* **limit** (*int**,* *optional*) – limit number of fetched records
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks



Returns:
pandas.DataFrame with listed experiments



Return type:
pandas.DataFrame




**Examples**



```
client.training.list()
training_runs_df = client.training.list(limit=100)
training_runs_df = client.training.list(limit=100, get_all=True)
training_runs_df = []
for entry in client.training.list(limit=100, asynchronous=True, get_all=True):
    training_runs_df.extend(entry)

```





list\_intermediate\_models(*training\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/training.html#Training.list_intermediate_models)[¶](#client.Training.list_intermediate_models "Link to this definition")
Print the intermediate\_models in a table format.



Parameters:
**training\_id** (*str*) – ID of the training





Note


This method is not supported for IBM Cloud Pak® for Data.



**Example**



```
client.training.list_intermediate_models()

```





monitor\_logs(*training\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/training.html#Training.monitor_logs)[¶](#client.Training.monitor_logs "Link to this definition")
Print the logs of a training created.



Parameters:
**training\_id** (*str*) – training ID





Note


This method is not supported for IBM Cloud Pak® for Data.



**Example**



```
client.training.monitor_logs(training_id)

```





monitor\_metrics(*training\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/training.html#Training.monitor_metrics)[¶](#client.Training.monitor_metrics "Link to this definition")
Print the metrics of a created training.



Parameters:
**training\_id** (*str*) – ID of the training





Note


This method is not supported for IBM Cloud Pak® for Data.



**Example**



```
client.training.monitor_metrics(training_id)

```





run(*meta\_props*, *asynchronous=True*)[[source]](_modules/ibm_watsonx_ai/training.html#Training.run)[¶](#client.Training.run "Link to this definition")
Create a new Machine Learning training.



Parameters:
* **meta\_props** (*dict*) – metadata of the training configuration. To see available meta names, use:



```
client.training.ConfigurationMetaNames.show()

```
* **asynchronous** (*bool**,* *optional*) – 
	+ True - training job is submitted and progress can be checked later
	+ False - method will wait till job completion and print training stats



Returns:
metadata of the training created



Return type:
dict





Note



You can provide one of the following values for training:* client.training.ConfigurationMetaNames.EXPERIMENT
* client.training.ConfigurationMetaNames.PIPELINE
* client.training.ConfigurationMetaNames.MODEL\_DEFINITION





**Examples**


Example of meta\_props for creating a training run in IBM Cloud Pak® for Data version 3.0.1 or above:



```
metadata = {
    client.training.ConfigurationMetaNames.NAME: 'Hand-written Digit Recognition',
    client.training.ConfigurationMetaNames.DESCRIPTION: 'Hand-written Digit Recognition Training',
    client.training.ConfigurationMetaNames.PIPELINE: {
        "id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab",
        "rev": "12",
        "model_type": "string",
        "data_bindings": [
            {
                "data_reference_name": "string",
                "node_id": "string"
            }
        ],
        "nodes_parameters": [
            {
                "node_id": "string",
                "parameters": {}
            }
        ],
        "hardware_spec": {
            "id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab",
            "rev": "12",
            "name": "string",
            "num_nodes": "2"
        }
    },
    client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [{
        'type': 's3',
        'connection': {},
        'location': {'href': 'v2/assets/asset1233456'},
        'schema': { 'id': 't1', 'name': 'Tasks', 'fields': [ { 'name': 'duration', 'type': 'number' } ]}
    }],
    client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
        'id' : 'string',
        'connection': {
            'endpoint_url': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
            'access_key_id': '***',
            'secret_access_key': '***'
        },
        'location': {
            'bucket': 'wml-dev-results',
            'path' : "path"
        }
        'type': 's3'
    }
}

```


Example of a Federated Learning training job:



```
aggregator_metadata = {
    client.training.ConfigurationMetaNames.NAME: 'Federated_Learning_Tensorflow_MNIST',
    client.training.ConfigurationMetaNames.DESCRIPTION: 'MNIST digit recognition with Federated Learning using Tensorflow',
    client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [],
    client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
        'type': results_type,
        'name': 'outputData',
        'connection': {},
        'location': { 'path': '/projects/' + PROJECT_ID + '/assets/trainings/'}
    },
    client.training.ConfigurationMetaNames.FEDERATED_LEARNING: {
        'model': {
            'type': 'tensorflow',
            'spec': {
            'id': untrained_model_id
        },
        'model_file': untrained_model_name
    },
    'fusion_type': 'iter_avg',
    'metrics': 'accuracy',
    'epochs': 3,
    'rounds': 10,
    'remote_training' : {
        'quorum': 1.0,
        'max_timeout': 3600,
        'remote_training_systems': [ { 'id': prime_rts_id }, { 'id': nonprime_rts_id} ]
    },
    'hardware_spec': {
        'name': 'S'
    },
    'software_spec': {
        'name': 'runtime-22.1-py3.9'
    }
}

aggregator = client.training.run(aggregator_metadata, asynchronous=True)
aggregator_id = client.training.get_id(aggregator)

```






*class* metanames.TrainingConfigurationMetaNames[[source]](_modules/metanames.html#TrainingConfigurationMetaNames)[¶](#metanames.TrainingConfigurationMetaNames "Link to this definition")
Set of MetaNames for trainings.


Available MetaNames:





| MetaName | Type | Required | Schema | Example value |
| --- | --- | --- | --- | --- |
| TRAINING\_DATA\_REFERENCES | list | Y | `[{'name(optional)': 'string', 'type(required)': 'string', 'connection(required)': {'endpoint_url(required)': 'string', 'access_key_id(required)': 'string', 'secret_access_key(required)': 'string'}, 'location(required)': {'bucket': 'string', 'path': 'string'}, 'schema(optional)': {'id(required)': 'string', 'fields(required)': [{'name(required)': 'string', 'type(required)': 'string', 'nullable(optional)': 'string'}]}}]` | `[{'connection': {'endpoint_url': 'https://s3-api.us-geo.objectstorage.softlayer.net', 'access_key_id': '***', 'secret_access_key': '***'}, 'location': {'bucket': 'train-data', 'path': 'training_path'}, 'type': 's3', 'schema': {'id': '1', 'fields': [{'name': 'x', 'type': 'double', 'nullable': 'False'}]}}]` |
| TRAINING\_RESULTS\_REFERENCE | dict | Y | `{'name(optional)': 'string', 'type(required)': 'string', 'connection(required)': {'endpoint_url(required)': 'string', 'access_key_id(required)': 'string', 'secret_access_key(required)': 'string'}, 'location(required)': {'bucket': 'string', 'path': 'string'}}` | `{'connection': {'endpoint_url': 'https://s3-api.us-geo.objectstorage.softlayer.net', 'access_key_id': '***', 'secret_access_key': '***'}, 'location': {'bucket': 'test-results', 'path': 'training_path'}, 'type': 's3'}` |
| TEST\_DATA\_REFERENCES | list | N | `[{'name(optional)': 'string', 'type(required)': 'string', 'connection(required)': {'endpoint_url(required)': 'string', 'access_key_id(required)': 'string', 'secret_access_key(required)': 'string'}, 'location(required)': {'bucket': 'string', 'path': 'string'}, 'schema(optional)': {'id(required)': 'string', 'fields(required)': [{'name(required)': 'string', 'type(required)': 'string', 'nullable(optional)': 'string'}]}}]` | `[{'connection': {'endpoint_url': 'https://s3-api.us-geo.objectstorage.softlayer.net', 'access_key_id': '***', 'secret_access_key': '***'}, 'location': {'bucket': 'train-data', 'path': 'training_path'}, 'type': 's3', 'schema': {'id': '1', 'fields': [{'name': 'x', 'type': 'double', 'nullable': 'False'}]}}]` |
| TEST\_OUTPUT\_DATA | dict | N | `{'name(optional)': 'string', 'type(required)': 'string', 'connection(required)': {'endpoint_url(required)': 'string', 'access_key_id(required)': 'string', 'secret_access_key(required)': 'string'}, 'location(required)': {'bucket': 'string', 'path': 'string'}, 'schema(optional)': {'id(required)': 'string', 'fields(required)': [{'name(required)': 'string', 'type(required)': 'string', 'nullable(optional)': 'string'}]}}` | `[{'connection': {'endpoint_url': 'https://s3-api.us-geo.objectstorage.softlayer.net', 'access_key_id': '***', 'secret_access_key': '***'}, 'location': {'bucket': 'train-data', 'path': 'training_path'}, 'type': 's3', 'schema': {'id': '1', 'fields': [{'name': 'x', 'type': 'double', 'nullable': 'False'}]}}]` |
| TAGS | list | N | `['string']` | `['string']` |
| PIPELINE | dict | N |  | `{'id': '3c1ce536-20dc-426e-aac7-7284cf3befc6', 'rev': '1', 'modeltype': 'tensorflow_1.1.3-py3', 'data_bindings': [{'data_reference_name': 'string', 'node_id': 'string'}], 'node_parameters': [{'node_id': 'string', 'parameters': {}}], 'hardware_spec': {'id': '4cedab6d-e8e4-4214-b81a-2ddb122db2ab', 'rev': '12', 'name': 'string', 'num_nodes': '2'}, 'hybrid_pipeline_hardware_specs': [{'node_runtime_id': 'string', 'hardware_spec': {'id': '4cedab6d-e8e4-4214-b81a-2ddb122db2ab', 'rev': '12', 'name': 'string', 'num_nodes': '2'}}]}` |
| EXPERIMENT | dict | N |  | `{'id': '3c1ce536-20dc-426e-aac7-7284cf3befc6', 'rev': 1, 'description': 'test experiment'}` |
| PROMPT\_TUNING | dict | N |  | `{'task_id': 'generation', 'base_model': {'name': 'google/flan-t5-xl'}}` |
| AUTO\_UPDATE\_MODEL | bool | N |  | `False` |
| FEDERATED\_LEARNING | dict | N |  | `3c1ce536-20dc-426e-aac7-7284cf3befc6` |
| SPACE\_UID | str | N |  | `3c1ce536-20dc-426e-aac7-7284cf3befc6` |
| MODEL\_DEFINITION | dict | N |  | `{'id': '4cedab6d-e8e4-4214-b81a-2ddb122db2ab', 'rev': '12', 'model_type': 'string', 'hardware_spec': {'id': '4cedab6d-e8e4-4214-b81a-2ddb122db2ab', 'rev': '12', 'name': 'string', 'num_nodes': '2'}, 'software_spec': {'id': '4cedab6d-e8e4-4214-b81a-2ddb122db2ab', 'rev': '12', 'name': '...'}, 'command': 'string', 'parameters': {}}` |
| DESCRIPTION | str | Y |  | `tensorflow model training` |
| NAME | str | Y |  | `sample training` |






Enums[¶](#module-ibm_watsonx_ai.utils.autoai.enums "Link to this heading")
--------------------------------------------------------------------------




*class* ibm\_watsonx\_ai.utils.autoai.enums.ClassificationAlgorithms(*value*)[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#ClassificationAlgorithms)[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms "Link to this definition")
Bases: `Enum`


Classification algorithms that AutoAI can use for IBM Cloud.




DT *= 'DecisionTreeClassifier'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.DT "Link to this definition")



EX\_TREES *= 'ExtraTreesClassifier'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.EX_TREES "Link to this definition")



GB *= 'GradientBoostingClassifier'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.GB "Link to this definition")



LGBM *= 'LGBMClassifier'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.LGBM "Link to this definition")



LR *= 'LogisticRegression'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.LR "Link to this definition")



RF *= 'RandomForestClassifier'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.RF "Link to this definition")



SnapBM *= 'SnapBoostingMachineClassifier'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.SnapBM "Link to this definition")



SnapDT *= 'SnapDecisionTreeClassifier'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.SnapDT "Link to this definition")



SnapLR *= 'SnapLogisticRegression'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.SnapLR "Link to this definition")



SnapRF *= 'SnapRandomForestClassifier'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.SnapRF "Link to this definition")



SnapSVM *= 'SnapSVMClassifier'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.SnapSVM "Link to this definition")



XGB *= 'XGBClassifier'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.XGB "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D(*value*)[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#ClassificationAlgorithmsCP4D)[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D "Link to this definition")
Bases: `Enum`


Classification algorithms that AutoAI can use for IBM Cloud Pak® for Data(CP4D).
The SnapML estimators (SnapDT, SnapRF, SnapSVM, SnapLR) are supported
on IBM Cloud Pak® for Data version 4.0.2 and above.




DT *= 'DecisionTreeClassifierEstimator'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.DT "Link to this definition")



EX\_TREES *= 'ExtraTreesClassifierEstimator'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.EX_TREES "Link to this definition")



GB *= 'GradientBoostingClassifierEstimator'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.GB "Link to this definition")



LGBM *= 'LGBMClassifierEstimator'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.LGBM "Link to this definition")



LR *= 'LogisticRegressionEstimator'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.LR "Link to this definition")



RF *= 'RandomForestClassifierEstimator'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.RF "Link to this definition")



SnapBM *= 'SnapBoostingMachineClassifier'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.SnapBM "Link to this definition")



SnapDT *= 'SnapDecisionTreeClassifier'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.SnapDT "Link to this definition")



SnapLR *= 'SnapLogisticRegression'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.SnapLR "Link to this definition")



SnapRF *= 'SnapRandomForestClassifier'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.SnapRF "Link to this definition")



SnapSVM *= 'SnapSVMClassifier'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.SnapSVM "Link to this definition")



XGB *= 'XGBClassifierEstimator'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.XGB "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.DataConnectionTypes[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#DataConnectionTypes)[¶](#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes "Link to this definition")
Bases: `object`


Supported types of DataConnection.




CA *= 'connection\_asset'*[¶](#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes.CA "Link to this definition")



CN *= 'container'*[¶](#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes.CN "Link to this definition")



DS *= 'data\_asset'*[¶](#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes.DS "Link to this definition")



FS *= 'fs'*[¶](#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes.FS "Link to this definition")



S3 *= 's3'*[¶](#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes.S3 "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.Directions[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#Directions)[¶](#ibm_watsonx_ai.utils.autoai.enums.Directions "Link to this definition")
Bases: `object`


Possible metrics directions




ASCENDING *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Directions.ASCENDING "Link to this definition")



DESCENDING *= 'descending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Directions.DESCENDING "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.ForecastingAlgorithms(*value*)[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#ForecastingAlgorithms)[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms "Link to this definition")
Bases: `Enum`


Forecasting algorithms that AutoAI can use for IBM watsonx.ai software with IBM Cloud Pak for Data.




ARIMA *= 'ARIMA'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.ARIMA "Link to this definition")



BATS *= 'BATS'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.BATS "Link to this definition")



ENSEMBLER *= 'Ensembler'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.ENSEMBLER "Link to this definition")



HW *= 'HoltWinters'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.HW "Link to this definition")



LR *= 'LinearRegression'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.LR "Link to this definition")



RF *= 'RandomForest'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.RF "Link to this definition")



SVM *= 'SVM'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.SVM "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D(*value*)[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#ForecastingAlgorithmsCP4D)[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D "Link to this definition")
Bases: `Enum`


Forecasting algorithms that AutoAI can use for IBM Cloud.




ARIMA *= 'ARIMA'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.ARIMA "Link to this definition")



BATS *= 'BATS'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.BATS "Link to this definition")



ENSEMBLER *= 'Ensembler'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.ENSEMBLER "Link to this definition")



HW *= 'HoltWinters'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.HW "Link to this definition")



LR *= 'LinearRegression'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.LR "Link to this definition")



RF *= 'RandomForest'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.RF "Link to this definition")



SVM *= 'SVM'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.SVM "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.ForecastingPipelineTypes(*value*)[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#ForecastingPipelineTypes)[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes "Link to this definition")
Bases: `Enum`


Forecasting pipeline types that AutoAI can use for IBM Cloud Pak® for Data(CP4D).




ARIMA *= 'ARIMA'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMA "Link to this definition")



ARIMAX *= 'ARIMAX'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMAX "Link to this definition")



ARIMAX\_DMLR *= 'ARIMAX\_DMLR'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMAX_DMLR "Link to this definition")



ARIMAX\_PALR *= 'ARIMAX\_PALR'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMAX_PALR "Link to this definition")



ARIMAX\_RAR *= 'ARIMAX\_RAR'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMAX_RAR "Link to this definition")



ARIMAX\_RSAR *= 'ARIMAX\_RSAR'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMAX_RSAR "Link to this definition")



Bats *= 'Bats'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.Bats "Link to this definition")



DifferenceFlattenEnsembler *= 'DifferenceFlattenEnsembler'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.DifferenceFlattenEnsembler "Link to this definition")



ExogenousDifferenceFlattenEnsembler *= 'ExogenousDifferenceFlattenEnsembler'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousDifferenceFlattenEnsembler "Link to this definition")



ExogenousFlattenEnsembler *= 'ExogenousFlattenEnsembler'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousFlattenEnsembler "Link to this definition")



ExogenousLocalizedFlattenEnsembler *= 'ExogenousLocalizedFlattenEnsembler'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousLocalizedFlattenEnsembler "Link to this definition")



ExogenousMT2RForecaster *= 'ExogenousMT2RForecaster'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousMT2RForecaster "Link to this definition")



ExogenousRandomForestRegressor *= 'ExogenousRandomForestRegressor'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousRandomForestRegressor "Link to this definition")



ExogenousSVM *= 'ExogenousSVM'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousSVM "Link to this definition")



FlattenEnsembler *= 'FlattenEnsembler'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.FlattenEnsembler "Link to this definition")



HoltWinterAdditive *= 'HoltWinterAdditive'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.HoltWinterAdditive "Link to this definition")



HoltWinterMultiplicative *= 'HoltWinterMultiplicative'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.HoltWinterMultiplicative "Link to this definition")



LocalizedFlattenEnsembler *= 'LocalizedFlattenEnsembler'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.LocalizedFlattenEnsembler "Link to this definition")



MT2RForecaster *= 'MT2RForecaster'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.MT2RForecaster "Link to this definition")



RandomForestRegressor *= 'RandomForestRegressor'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.RandomForestRegressor "Link to this definition")



SVM *= 'SVM'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.SVM "Link to this definition")



*static* get\_exogenous()[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#ForecastingPipelineTypes.get_exogenous)[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.get_exogenous "Link to this definition")
Get a list of pipelines that use supporting features (exogenous pipelines).



Returns:
list of pipelines using supporting features



Return type:
list[[ForecastingPipelineTypes](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes "ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes")]







*static* get\_non\_exogenous()[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#ForecastingPipelineTypes.get_non_exogenous)[¶](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.get_non_exogenous "Link to this definition")
Get a list of pipelines that are not using supporting features (non-exogenous pipelines).



Returns:
list of pipelines that do not use supporting features



Return type:
list[[ForecastingPipelineTypes](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes "ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes")]








*class* ibm\_watsonx\_ai.utils.autoai.enums.ImputationStrategy(*value*)[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#ImputationStrategy)[¶](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy "Link to this definition")
Bases: `Enum`


Missing values imputation strategies.




BEST\_OF\_DEFAULT\_IMPUTERS *= 'best\_of\_default\_imputers'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.BEST_OF_DEFAULT_IMPUTERS "Link to this definition")



CUBIC *= 'cubic'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.CUBIC "Link to this definition")



FLATTEN\_ITERATIVE *= 'flatten\_iterative'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.FLATTEN_ITERATIVE "Link to this definition")



LINEAR *= 'linear'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.LINEAR "Link to this definition")



MEAN *= 'mean'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.MEAN "Link to this definition")



MEDIAN *= 'median'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.MEDIAN "Link to this definition")



MOST\_FREQUENT *= 'most\_frequent'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.MOST_FREQUENT "Link to this definition")



NEXT *= 'next'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.NEXT "Link to this definition")



NO\_IMPUTATION *= 'no\_imputation'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.NO_IMPUTATION "Link to this definition")



PREVIOUS *= 'previous'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.PREVIOUS "Link to this definition")



VALUE *= 'value'*[¶](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.VALUE "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.Metrics[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#Metrics)[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics "Link to this definition")
Bases: `object`


Supported types of classification and regression metrics in AutoAI.




ACCURACY\_AND\_DISPARATE\_IMPACT\_SCORE *= 'accuracy\_and\_disparate\_impact'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.ACCURACY_AND_DISPARATE_IMPACT_SCORE "Link to this definition")



ACCURACY\_SCORE *= 'accuracy'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.ACCURACY_SCORE "Link to this definition")



AVERAGE\_PRECISION\_SCORE *= 'average\_precision'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.AVERAGE_PRECISION_SCORE "Link to this definition")



EXPLAINED\_VARIANCE\_SCORE *= 'explained\_variance'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.EXPLAINED_VARIANCE_SCORE "Link to this definition")



F1\_SCORE *= 'f1'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.F1_SCORE "Link to this definition")



F1\_SCORE\_MACRO *= 'f1\_macro'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.F1_SCORE_MACRO "Link to this definition")



F1\_SCORE\_MICRO *= 'f1\_micro'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.F1_SCORE_MICRO "Link to this definition")



F1\_SCORE\_WEIGHTED *= 'f1\_weighted'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.F1_SCORE_WEIGHTED "Link to this definition")



LOG\_LOSS *= 'neg\_log\_loss'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.LOG_LOSS "Link to this definition")



MEAN\_ABSOLUTE\_ERROR *= 'neg\_mean\_absolute\_error'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.MEAN_ABSOLUTE_ERROR "Link to this definition")



MEAN\_SQUARED\_ERROR *= 'neg\_mean\_squared\_error'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.MEAN_SQUARED_ERROR "Link to this definition")



MEAN\_SQUARED\_LOG\_ERROR *= 'neg\_mean\_squared\_log\_error'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.MEAN_SQUARED_LOG_ERROR "Link to this definition")



MEDIAN\_ABSOLUTE\_ERROR *= 'neg\_median\_absolute\_error'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.MEDIAN_ABSOLUTE_ERROR "Link to this definition")



PRECISION\_SCORE *= 'precision'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.PRECISION_SCORE "Link to this definition")



PRECISION\_SCORE\_MACRO *= 'precision\_macro'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.PRECISION_SCORE_MACRO "Link to this definition")



PRECISION\_SCORE\_MICRO *= 'precision\_micro'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.PRECISION_SCORE_MICRO "Link to this definition")



PRECISION\_SCORE\_WEIGHTED *= 'precision\_weighted'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.PRECISION_SCORE_WEIGHTED "Link to this definition")



R2\_AND\_DISPARATE\_IMPACT\_SCORE *= 'r2\_and\_disparate\_impact'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.R2_AND_DISPARATE_IMPACT_SCORE "Link to this definition")



R2\_SCORE *= 'r2'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.R2_SCORE "Link to this definition")



RECALL\_SCORE *= 'recall'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.RECALL_SCORE "Link to this definition")



RECALL\_SCORE\_MACRO *= 'recall\_macro'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.RECALL_SCORE_MACRO "Link to this definition")



RECALL\_SCORE\_MICRO *= 'recall\_micro'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.RECALL_SCORE_MICRO "Link to this definition")



RECALL\_SCORE\_WEIGHTED *= 'recall\_weighted'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.RECALL_SCORE_WEIGHTED "Link to this definition")



ROC\_AUC\_SCORE *= 'roc\_auc'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.ROC_AUC_SCORE "Link to this definition")



ROOT\_MEAN\_SQUARED\_ERROR *= 'neg\_root\_mean\_squared\_error'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.ROOT_MEAN_SQUARED_ERROR "Link to this definition")



ROOT\_MEAN\_SQUARED\_LOG\_ERROR *= 'neg\_root\_mean\_squared\_log\_error'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Metrics.ROOT_MEAN_SQUARED_LOG_ERROR "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.MetricsToDirections(*value*)[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#MetricsToDirections)[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections "Link to this definition")
Bases: `Enum`


Map of metrics directions.




ACCURACY *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.ACCURACY "Link to this definition")



AVERAGE\_PRECISION *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.AVERAGE_PRECISION "Link to this definition")



EXPLAINED\_VARIANCE *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.EXPLAINED_VARIANCE "Link to this definition")



F1 *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.F1 "Link to this definition")



F1\_MACRO *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.F1_MACRO "Link to this definition")



F1\_MICRO *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.F1_MICRO "Link to this definition")



F1\_WEIGHTED *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.F1_WEIGHTED "Link to this definition")



NEG\_LOG\_LOSS *= 'descending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_LOG_LOSS "Link to this definition")



NEG\_MEAN\_ABSOLUTE\_ERROR *= 'descending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_MEAN_ABSOLUTE_ERROR "Link to this definition")



NEG\_MEAN\_SQUARED\_ERROR *= 'descending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_MEAN_SQUARED_ERROR "Link to this definition")



NEG\_MEAN\_SQUARED\_LOG\_ERROR *= 'descending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_MEAN_SQUARED_LOG_ERROR "Link to this definition")



NEG\_MEDIAN\_ABSOLUTE\_ERROR *= 'descending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_MEDIAN_ABSOLUTE_ERROR "Link to this definition")



NEG\_ROOT\_MEAN\_SQUARED\_ERROR *= 'descending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_ROOT_MEAN_SQUARED_ERROR "Link to this definition")



NEG\_ROOT\_MEAN\_SQUARED\_LOG\_ERROR *= 'descending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_ROOT_MEAN_SQUARED_LOG_ERROR "Link to this definition")



NORMALIZED\_GINI\_COEFFICIENT *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NORMALIZED_GINI_COEFFICIENT "Link to this definition")



PRECISION *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.PRECISION "Link to this definition")



PRECISION\_MACRO *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.PRECISION_MACRO "Link to this definition")



PRECISION\_MICRO *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.PRECISION_MICRO "Link to this definition")



PRECISION\_WEIGHTED *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.PRECISION_WEIGHTED "Link to this definition")



R2 *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.R2 "Link to this definition")



RECALL *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.RECALL "Link to this definition")



RECALL\_MACRO *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.RECALL_MACRO "Link to this definition")



RECALL\_MICRO *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.RECALL_MICRO "Link to this definition")



RECALL\_WEIGHTED *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.RECALL_WEIGHTED "Link to this definition")



ROC\_AUC *= 'ascending'*[¶](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.ROC_AUC "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.PipelineTypes[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#PipelineTypes)[¶](#ibm_watsonx_ai.utils.autoai.enums.PipelineTypes "Link to this definition")
Bases: `object`


Supported types of Pipelines.




LALE *= 'lale'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PipelineTypes.LALE "Link to this definition")



SKLEARN *= 'sklearn'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PipelineTypes.SKLEARN "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.PositiveLabelClass[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#PositiveLabelClass)[¶](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass "Link to this definition")
Bases: `object`


Metrics that need positive label definition for binary classification.




AVERAGE\_PRECISION\_SCORE *= 'average\_precision'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.AVERAGE_PRECISION_SCORE "Link to this definition")



F1\_SCORE *= 'f1'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.F1_SCORE "Link to this definition")



F1\_SCORE\_MACRO *= 'f1\_macro'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.F1_SCORE_MACRO "Link to this definition")



F1\_SCORE\_MICRO *= 'f1\_micro'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.F1_SCORE_MICRO "Link to this definition")



F1\_SCORE\_WEIGHTED *= 'f1\_weighted'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.F1_SCORE_WEIGHTED "Link to this definition")



PRECISION\_SCORE *= 'precision'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.PRECISION_SCORE "Link to this definition")



PRECISION\_SCORE\_MACRO *= 'precision\_macro'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.PRECISION_SCORE_MACRO "Link to this definition")



PRECISION\_SCORE\_MICRO *= 'precision\_micro'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.PRECISION_SCORE_MICRO "Link to this definition")



PRECISION\_SCORE\_WEIGHTED *= 'precision\_weighted'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.PRECISION_SCORE_WEIGHTED "Link to this definition")



RECALL\_SCORE *= 'recall'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.RECALL_SCORE "Link to this definition")



RECALL\_SCORE\_MACRO *= 'recall\_macro'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.RECALL_SCORE_MACRO "Link to this definition")



RECALL\_SCORE\_MICRO *= 'recall\_micro'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.RECALL_SCORE_MICRO "Link to this definition")



RECALL\_SCORE\_WEIGHTED *= 'recall\_weighted'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.RECALL_SCORE_WEIGHTED "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.PredictionType[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#PredictionType)[¶](#ibm_watsonx_ai.utils.autoai.enums.PredictionType "Link to this definition")
Bases: `object`


Supported types of learning.




BINARY *= 'binary'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PredictionType.BINARY "Link to this definition")



CLASSIFICATION *= 'classification'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PredictionType.CLASSIFICATION "Link to this definition")



FORECASTING *= 'forecasting'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PredictionType.FORECASTING "Link to this definition")



MULTICLASS *= 'multiclass'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PredictionType.MULTICLASS "Link to this definition")



REGRESSION *= 'regression'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PredictionType.REGRESSION "Link to this definition")



TIMESERIES\_ANOMALY\_PREDICTION *= 'timeseries\_anomaly\_prediction'*[¶](#ibm_watsonx_ai.utils.autoai.enums.PredictionType.TIMESERIES_ANOMALY_PREDICTION "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.RegressionAlgorithms(*value*)[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#RegressionAlgorithms)[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms "Link to this definition")
Bases: `Enum`


Regression algorithms that AutoAI can use for IBM Cloud.




DT *= 'DecisionTreeRegressor'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.DT "Link to this definition")



EX\_TREES *= 'ExtraTreesRegressor'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.EX_TREES "Link to this definition")



GB *= 'GradientBoostingRegressor'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.GB "Link to this definition")



LGBM *= 'LGBMRegressor'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.LGBM "Link to this definition")



LR *= 'LinearRegression'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.LR "Link to this definition")



RF *= 'RandomForestRegressor'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.RF "Link to this definition")



RIDGE *= 'Ridge'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.RIDGE "Link to this definition")



SnapBM *= 'SnapBoostingMachineRegressor'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.SnapBM "Link to this definition")



SnapDT *= 'SnapDecisionTreeRegressor'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.SnapDT "Link to this definition")



SnapRF *= 'SnapRandomForestRegressor'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.SnapRF "Link to this definition")



XGB *= 'XGBRegressor'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.XGB "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.RegressionAlgorithmsCP4D(*value*)[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#RegressionAlgorithmsCP4D)[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D "Link to this definition")
Bases: `Enum`


Regression algorithms that AutoAI can use for IBM Cloud Pak® for Data(CP4D).
The SnapML estimators (SnapDT, SnapRF, SnapBM) are supported
on IBM Cloud Pak® for Data version 4.0.2 and above.




DT *= 'DecisionTreeRegressorEstimator'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.DT "Link to this definition")



EX\_TREES *= 'ExtraTreesRegressorEstimator'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.EX_TREES "Link to this definition")



GB *= 'GradientBoostingRegressorEstimator'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.GB "Link to this definition")



LGBM *= 'LGBMRegressorEstimator'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.LGBM "Link to this definition")



LR *= 'LinearRegressionEstimator'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.LR "Link to this definition")



RF *= 'RandomForestRegressorEstimator'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.RF "Link to this definition")



RIDGE *= 'RidgeEstimator'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.RIDGE "Link to this definition")



SnapBM *= 'SnapBoostingMachineRegressor'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.SnapBM "Link to this definition")



SnapDT *= 'SnapDecisionTreeRegressor'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.SnapDT "Link to this definition")



SnapRF *= 'SnapRandomForestRegressor'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.SnapRF "Link to this definition")



XGB *= 'XGBRegressorEstimator'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.XGB "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.RunStateTypes[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#RunStateTypes)[¶](#ibm_watsonx_ai.utils.autoai.enums.RunStateTypes "Link to this definition")
Bases: `object`


Supported types of AutoAI fit/run.




COMPLETED *= 'completed'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RunStateTypes.COMPLETED "Link to this definition")



FAILED *= 'failed'*[¶](#ibm_watsonx_ai.utils.autoai.enums.RunStateTypes.FAILED "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.SamplingTypes[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#SamplingTypes)[¶](#ibm_watsonx_ai.utils.autoai.enums.SamplingTypes "Link to this definition")
Bases: `object`


Types of training data sampling.




FIRST\_VALUES *= 'first\_n\_records'*[¶](#ibm_watsonx_ai.utils.autoai.enums.SamplingTypes.FIRST_VALUES "Link to this definition")



LAST\_VALUES *= 'truncate'*[¶](#ibm_watsonx_ai.utils.autoai.enums.SamplingTypes.LAST_VALUES "Link to this definition")



RANDOM *= 'random'*[¶](#ibm_watsonx_ai.utils.autoai.enums.SamplingTypes.RANDOM "Link to this definition")



STRATIFIED *= 'stratified'*[¶](#ibm_watsonx_ai.utils.autoai.enums.SamplingTypes.STRATIFIED "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.TShirtSize[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#TShirtSize)[¶](#ibm_watsonx_ai.utils.autoai.enums.TShirtSize "Link to this definition")
Bases: `object`


Possible sizes of the AutoAI POD.
Depending on the POD size, AutoAI can support different data set sizes.


* S - small (2vCPUs and 8GB of RAM)
* M - Medium (4vCPUs and 16GB of RAM)
* L - Large (8vCPUs and 32GB of RAM))
* XL - Extra Large (16vCPUs and 64GB of RAM)




L *= 'l'*[¶](#ibm_watsonx_ai.utils.autoai.enums.TShirtSize.L "Link to this definition")



M *= 'm'*[¶](#ibm_watsonx_ai.utils.autoai.enums.TShirtSize.M "Link to this definition")



S *= 's'*[¶](#ibm_watsonx_ai.utils.autoai.enums.TShirtSize.S "Link to this definition")



XL *= 'xl'*[¶](#ibm_watsonx_ai.utils.autoai.enums.TShirtSize.XL "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.TimeseriesAnomalyPredictionAlgorithms(*value*)[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#TimeseriesAnomalyPredictionAlgorithms)[¶](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionAlgorithms "Link to this definition")
Bases: `Enum`


Timeseries Anomaly Prediction algorithms that AutoAI can use for IBM Cloud.




Forecasting *= 'Forecasting'*[¶](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionAlgorithms.Forecasting "Link to this definition")



Relationship *= 'Relationship'*[¶](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionAlgorithms.Relationship "Link to this definition")



Window *= 'Window'*[¶](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionAlgorithms.Window "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes(*value*)[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#TimeseriesAnomalyPredictionPipelineTypes)[¶](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes "Link to this definition")
Bases: `Enum`


Timeseries Anomaly Prediction pipeline types that AutoAI can use for IBM Cloud.




PointwiseBoundedBATS *= 'PointwiseBoundedBATS'*[¶](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.PointwiseBoundedBATS "Link to this definition")



PointwiseBoundedBATSForceUpdate *= 'PointwiseBoundedBATSForceUpdate'*[¶](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.PointwiseBoundedBATSForceUpdate "Link to this definition")



PointwiseBoundedHoltWintersAdditive *= 'PointwiseBoundedHoltWintersAdditive'*[¶](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.PointwiseBoundedHoltWintersAdditive "Link to this definition")



WindowLOF *= 'WindowLOF'*[¶](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.WindowLOF "Link to this definition")



WindowNN *= 'WindowNN'*[¶](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.WindowNN "Link to this definition")



WindowPCA *= 'WindowPCA'*[¶](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.WindowPCA "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.Transformers[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#Transformers)[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers "Link to this definition")
Bases: `object`


Supported types of congito transformers names in AutoAI.




ABS *= 'abs'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.ABS "Link to this definition")



CBRT *= 'cbrt'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.CBRT "Link to this definition")



COS *= 'cos'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.COS "Link to this definition")



CUBE *= 'cube'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.CUBE "Link to this definition")



DIFF *= 'diff'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.DIFF "Link to this definition")



DIVIDE *= 'divide'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.DIVIDE "Link to this definition")



FEATUREAGGLOMERATION *= 'featureagglomeration'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.FEATUREAGGLOMERATION "Link to this definition")



ISOFORESTANOMALY *= 'isoforestanomaly'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.ISOFORESTANOMALY "Link to this definition")



LOG *= 'log'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.LOG "Link to this definition")



MAX *= 'max'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.MAX "Link to this definition")



MINMAXSCALER *= 'minmaxscaler'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.MINMAXSCALER "Link to this definition")



NXOR *= 'nxor'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.NXOR "Link to this definition")



PCA *= 'pca'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.PCA "Link to this definition")



PRODUCT *= 'product'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.PRODUCT "Link to this definition")



ROUND *= 'round'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.ROUND "Link to this definition")



SIGMOID *= 'sigmoid'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.SIGMOID "Link to this definition")



SIN *= 'sin'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.SIN "Link to this definition")



SQRT *= 'sqrt'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.SQRT "Link to this definition")



SQUARE *= 'square'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.SQUARE "Link to this definition")



STDSCALER *= 'stdscaler'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.STDSCALER "Link to this definition")



SUM *= 'sum'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.SUM "Link to this definition")



TAN *= 'tan'*[¶](#ibm_watsonx_ai.utils.autoai.enums.Transformers.TAN "Link to this definition")




*class* ibm\_watsonx\_ai.utils.autoai.enums.VisualizationTypes[[source]](_modules/ibm_watsonx_ai/utils/autoai/enums.html#VisualizationTypes)[¶](#ibm_watsonx_ai.utils.autoai.enums.VisualizationTypes "Link to this definition")
Bases: `object`


Types of visualization options.




INPLACE *= 'inplace'*[¶](#ibm_watsonx_ai.utils.autoai.enums.VisualizationTypes.INPLACE "Link to this definition")



PDF *= 'pdf'*[¶](#ibm_watsonx_ai.utils.autoai.enums.VisualizationTypes.PDF "Link to this definition")








[Next

Federated Learning](federated_learning.html)
[Previous

Base](base.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Core](#)
	+ [Connections](#connections)
		- [`Connections`](#client.Connections)
			* [`Connections.ConfigurationMetaNames`](#client.Connections.ConfigurationMetaNames)
			* [`Connections.create()`](#client.Connections.create)
			* [`Connections.delete()`](#client.Connections.delete)
			* [`Connections.get_datasource_type_details_by_id()`](#client.Connections.get_datasource_type_details_by_id)
			* [`Connections.get_datasource_type_id_by_name()`](#client.Connections.get_datasource_type_id_by_name)
			* [`Connections.get_datasource_type_uid_by_name()`](#client.Connections.get_datasource_type_uid_by_name)
			* [`Connections.get_details()`](#client.Connections.get_details)
			* [`Connections.get_id()`](#client.Connections.get_id)
			* [`Connections.get_uid()`](#client.Connections.get_uid)
			* [`Connections.get_uploaded_db_drivers()`](#client.Connections.get_uploaded_db_drivers)
			* [`Connections.list()`](#client.Connections.list)
			* [`Connections.list_datasource_types()`](#client.Connections.list_datasource_types)
			* [`Connections.list_uploaded_db_drivers()`](#client.Connections.list_uploaded_db_drivers)
			* [`Connections.sign_db_driver_url()`](#client.Connections.sign_db_driver_url)
			* [`Connections.upload_db_driver()`](#client.Connections.upload_db_driver)
		- [`ConnectionMetaNames`](#metanames.ConnectionMetaNames)
	+ [Data assets](#data-assets)
		- [`Assets`](#client.Assets)
			* [`Assets.ConfigurationMetaNames`](#client.Assets.ConfigurationMetaNames)
			* [`Assets.create()`](#client.Assets.create)
			* [`Assets.delete()`](#client.Assets.delete)
			* [`Assets.download()`](#client.Assets.download)
			* [`Assets.get_content()`](#client.Assets.get_content)
			* [`Assets.get_details()`](#client.Assets.get_details)
			* [`Assets.get_href()`](#client.Assets.get_href)
			* [`Assets.get_id()`](#client.Assets.get_id)
			* [`Assets.list()`](#client.Assets.list)
			* [`Assets.store()`](#client.Assets.store)
		- [`AssetsMetaNames`](#metanames.AssetsMetaNames)
	+ [Deployments](#deployments)
		- [`Deployments`](#client.Deployments)
			* [`Deployments.create()`](#client.Deployments.create)
			* [`Deployments.create_job()`](#client.Deployments.create_job)
			* [`Deployments.delete()`](#client.Deployments.delete)
			* [`Deployments.delete_job()`](#client.Deployments.delete_job)
			* [`Deployments.generate()`](#client.Deployments.generate)
			* [`Deployments.generate_text()`](#client.Deployments.generate_text)
			* [`Deployments.generate_text_stream()`](#client.Deployments.generate_text_stream)
			* [`Deployments.get_details()`](#client.Deployments.get_details)
			* [`Deployments.get_download_url()`](#client.Deployments.get_download_url)
			* [`Deployments.get_href()`](#client.Deployments.get_href)
			* [`Deployments.get_id()`](#client.Deployments.get_id)
			* [`Deployments.get_job_details()`](#client.Deployments.get_job_details)
			* [`Deployments.get_job_href()`](#client.Deployments.get_job_href)
			* [`Deployments.get_job_id()`](#client.Deployments.get_job_id)
			* [`Deployments.get_job_status()`](#client.Deployments.get_job_status)
			* [`Deployments.get_job_uid()`](#client.Deployments.get_job_uid)
			* [`Deployments.get_scoring_href()`](#client.Deployments.get_scoring_href)
			* [`Deployments.get_serving_href()`](#client.Deployments.get_serving_href)
			* [`Deployments.get_uid()`](#client.Deployments.get_uid)
			* [`Deployments.is_serving_name_available()`](#client.Deployments.is_serving_name_available)
			* [`Deployments.list()`](#client.Deployments.list)
			* [`Deployments.list_jobs()`](#client.Deployments.list_jobs)
			* [`Deployments.score()`](#client.Deployments.score)
			* [`Deployments.update()`](#client.Deployments.update)
		- [`DeploymentMetaNames`](#metanames.DeploymentMetaNames)
		- [`RShinyAuthenticationValues`](#ibm_watsonx_ai.utils.enums.RShinyAuthenticationValues)
			* [`RShinyAuthenticationValues.ANYONE_WITH_URL`](#ibm_watsonx_ai.utils.enums.RShinyAuthenticationValues.ANYONE_WITH_URL)
			* [`RShinyAuthenticationValues.ANY_VALID_USER`](#ibm_watsonx_ai.utils.enums.RShinyAuthenticationValues.ANY_VALID_USER)
			* [`RShinyAuthenticationValues.MEMBERS_OF_DEPLOYMENT_SPACE`](#ibm_watsonx_ai.utils.enums.RShinyAuthenticationValues.MEMBERS_OF_DEPLOYMENT_SPACE)
		- [`ScoringMetaNames`](#metanames.ScoringMetaNames)
		- [`DecisionOptimizationMetaNames`](#metanames.DecisionOptimizationMetaNames)
	+ [Export/Import](#export-import)
		- [`Export`](#client.Export)
			* [`Export.cancel()`](#client.Export.cancel)
			* [`Export.delete()`](#client.Export.delete)
			* [`Export.get_details()`](#client.Export.get_details)
			* [`Export.get_exported_content()`](#client.Export.get_exported_content)
			* [`Export.get_id()`](#client.Export.get_id)
			* [`Export.list()`](#client.Export.list)
			* [`Export.start()`](#client.Export.start)
		- [`Import`](#client.Import)
			* [`Import.cancel()`](#client.Import.cancel)
			* [`Import.delete()`](#client.Import.delete)
			* [`Import.get_details()`](#client.Import.get_details)
			* [`Import.get_id()`](#client.Import.get_id)
			* [`Import.list()`](#client.Import.list)
			* [`Import.start()`](#client.Import.start)
	+ [Factsheets (IBM Cloud only)](#factsheets-ibm-cloud-only)
		- [`Factsheets`](#client.Factsheets)
			* [`Factsheets.list_model_entries()`](#client.Factsheets.list_model_entries)
			* [`Factsheets.register_model_entry()`](#client.Factsheets.register_model_entry)
			* [`Factsheets.unregister_model_entry()`](#client.Factsheets.unregister_model_entry)
		- [`FactsheetsMetaNames`](#metanames.FactsheetsMetaNames)
	+ [Hardware specifications](#hardware-specifications)
		- [`HwSpec`](#client.HwSpec)
			* [`HwSpec.ConfigurationMetaNames`](#client.HwSpec.ConfigurationMetaNames)
			* [`HwSpec.delete()`](#client.HwSpec.delete)
			* [`HwSpec.get_details()`](#client.HwSpec.get_details)
			* [`HwSpec.get_href()`](#client.HwSpec.get_href)
			* [`HwSpec.get_id()`](#client.HwSpec.get_id)
			* [`HwSpec.get_id_by_name()`](#client.HwSpec.get_id_by_name)
			* [`HwSpec.get_uid()`](#client.HwSpec.get_uid)
			* [`HwSpec.get_uid_by_name()`](#client.HwSpec.get_uid_by_name)
			* [`HwSpec.list()`](#client.HwSpec.list)
			* [`HwSpec.store()`](#client.HwSpec.store)
		- [`HwSpecMetaNames`](#metanames.HwSpecMetaNames)
	+ [Helpers](#helpers)
		- [`get_credentials_from_config`](#ibm_watsonx_ai.helpers.helpers.get_credentials_from_config)
	+ [Model definitions](#model-definitions)
		- [`ModelDefinition`](#client.ModelDefinition)
			* [`ModelDefinition.ConfigurationMetaNames`](#client.ModelDefinition.ConfigurationMetaNames)
			* [`ModelDefinition.create_revision()`](#client.ModelDefinition.create_revision)
			* [`ModelDefinition.delete()`](#client.ModelDefinition.delete)
			* [`ModelDefinition.download()`](#client.ModelDefinition.download)
			* [`ModelDefinition.get_details()`](#client.ModelDefinition.get_details)
			* [`ModelDefinition.get_href()`](#client.ModelDefinition.get_href)
			* [`ModelDefinition.get_id()`](#client.ModelDefinition.get_id)
			* [`ModelDefinition.get_revision_details()`](#client.ModelDefinition.get_revision_details)
			* [`ModelDefinition.get_uid()`](#client.ModelDefinition.get_uid)
			* [`ModelDefinition.list()`](#client.ModelDefinition.list)
			* [`ModelDefinition.list_revisions()`](#client.ModelDefinition.list_revisions)
			* [`ModelDefinition.store()`](#client.ModelDefinition.store)
			* [`ModelDefinition.update()`](#client.ModelDefinition.update)
		- [`ModelDefinitionMetaNames`](#metanames.ModelDefinitionMetaNames)
	+ [Package extensions](#package-extensions)
		- [`PkgExtn`](#client.PkgExtn)
			* [`PkgExtn.ConfigurationMetaNames`](#client.PkgExtn.ConfigurationMetaNames)
			* [`PkgExtn.delete()`](#client.PkgExtn.delete)
			* [`PkgExtn.download()`](#client.PkgExtn.download)
			* [`PkgExtn.get_details()`](#client.PkgExtn.get_details)
			* [`PkgExtn.get_href()`](#client.PkgExtn.get_href)
			* [`PkgExtn.get_id()`](#client.PkgExtn.get_id)
			* [`PkgExtn.get_id_by_name()`](#client.PkgExtn.get_id_by_name)
			* [`PkgExtn.list()`](#client.PkgExtn.list)
			* [`PkgExtn.store()`](#client.PkgExtn.store)
		- [`PkgExtnMetaNames`](#metanames.PkgExtnMetaNames)
	+ [Parameter Sets](#parameter-sets)
		- [`ParameterSets`](#client.ParameterSets)
			* [`ParameterSets.ConfigurationMetaNames`](#client.ParameterSets.ConfigurationMetaNames)
			* [`ParameterSets.create()`](#client.ParameterSets.create)
			* [`ParameterSets.delete()`](#client.ParameterSets.delete)
			* [`ParameterSets.get_details()`](#client.ParameterSets.get_details)
			* [`ParameterSets.get_id_by_name()`](#client.ParameterSets.get_id_by_name)
			* [`ParameterSets.list()`](#client.ParameterSets.list)
			* [`ParameterSets.update()`](#client.ParameterSets.update)
		- [`ParameterSetsMetaNames`](#metanames.ParameterSetsMetaNames)
	+ [Repository](#repository)
		- [`Repository`](#client.Repository)
			* [`Repository.ModelAssetTypes`](#client.Repository.ModelAssetTypes)
			* [`Repository.create_experiment_revision()`](#client.Repository.create_experiment_revision)
			* [`Repository.create_function_revision()`](#client.Repository.create_function_revision)
			* [`Repository.create_model_revision()`](#client.Repository.create_model_revision)
			* [`Repository.create_pipeline_revision()`](#client.Repository.create_pipeline_revision)
			* [`Repository.create_revision()`](#client.Repository.create_revision)
			* [`Repository.delete()`](#client.Repository.delete)
			* [`Repository.download()`](#client.Repository.download)
			* [`Repository.get_details()`](#client.Repository.get_details)
			* [`Repository.get_experiment_details()`](#client.Repository.get_experiment_details)
			* [`Repository.get_experiment_href()`](#client.Repository.get_experiment_href)
			* [`Repository.get_experiment_id()`](#client.Repository.get_experiment_id)
			* [`Repository.get_experiment_revision_details()`](#client.Repository.get_experiment_revision_details)
			* [`Repository.get_function_details()`](#client.Repository.get_function_details)
			* [`Repository.get_function_href()`](#client.Repository.get_function_href)
			* [`Repository.get_function_id()`](#client.Repository.get_function_id)
			* [`Repository.get_function_revision_details()`](#client.Repository.get_function_revision_details)
			* [`Repository.get_model_details()`](#client.Repository.get_model_details)
			* [`Repository.get_model_href()`](#client.Repository.get_model_href)
			* [`Repository.get_model_id()`](#client.Repository.get_model_id)
			* [`Repository.get_model_revision_details()`](#client.Repository.get_model_revision_details)
			* [`Repository.get_pipeline_details()`](#client.Repository.get_pipeline_details)
			* [`Repository.get_pipeline_href()`](#client.Repository.get_pipeline_href)
			* [`Repository.get_pipeline_id()`](#client.Repository.get_pipeline_id)
			* [`Repository.get_pipeline_revision_details()`](#client.Repository.get_pipeline_revision_details)
			* [`Repository.list()`](#client.Repository.list)
			* [`Repository.list_experiments()`](#client.Repository.list_experiments)
			* [`Repository.list_experiments_revisions()`](#client.Repository.list_experiments_revisions)
			* [`Repository.list_functions()`](#client.Repository.list_functions)
			* [`Repository.list_functions_revisions()`](#client.Repository.list_functions_revisions)
			* [`Repository.list_models()`](#client.Repository.list_models)
			* [`Repository.list_models_revisions()`](#client.Repository.list_models_revisions)
			* [`Repository.list_pipelines()`](#client.Repository.list_pipelines)
			* [`Repository.list_pipelines_revisions()`](#client.Repository.list_pipelines_revisions)
			* [`Repository.load()`](#client.Repository.load)
			* [`Repository.promote_model()`](#client.Repository.promote_model)
			* [`Repository.store_experiment()`](#client.Repository.store_experiment)
			* [`Repository.store_function()`](#client.Repository.store_function)
			* [`Repository.store_model()`](#client.Repository.store_model)
			* [`Repository.store_pipeline()`](#client.Repository.store_pipeline)
			* [`Repository.update_experiment()`](#client.Repository.update_experiment)
			* [`Repository.update_function()`](#client.Repository.update_function)
			* [`Repository.update_model()`](#client.Repository.update_model)
			* [`Repository.update_pipeline()`](#client.Repository.update_pipeline)
		- [`ModelMetaNames`](#metanames.ModelMetaNames)
		- [`ExperimentMetaNames`](#metanames.ExperimentMetaNames)
		- [`FunctionMetaNames`](#metanames.FunctionMetaNames)
		- [`PipelineMetanames`](#metanames.PipelineMetanames)
	+ [Script](#script)
		- [`Script`](#client.Script)
			* [`Script.ConfigurationMetaNames`](#client.Script.ConfigurationMetaNames)
			* [`Script.create_revision()`](#client.Script.create_revision)
			* [`Script.delete()`](#client.Script.delete)
			* [`Script.download()`](#client.Script.download)
			* [`Script.get_details()`](#client.Script.get_details)
			* [`Script.get_href()`](#client.Script.get_href)
			* [`Script.get_id()`](#client.Script.get_id)
			* [`Script.get_revision_details()`](#client.Script.get_revision_details)
			* [`Script.list()`](#client.Script.list)
			* [`Script.list_revisions()`](#client.Script.list_revisions)
			* [`Script.store()`](#client.Script.store)
			* [`Script.update()`](#client.Script.update)
		- [`ScriptMetaNames`](#metanames.ScriptMetaNames)
	+ [Service instance](#service-instance)
		- [`ServiceInstance`](#client.ServiceInstance)
			* [`ServiceInstance.get_api_key()`](#client.ServiceInstance.get_api_key)
			* [`ServiceInstance.get_details()`](#client.ServiceInstance.get_details)
			* [`ServiceInstance.get_instance_id()`](#client.ServiceInstance.get_instance_id)
			* [`ServiceInstance.get_password()`](#client.ServiceInstance.get_password)
			* [`ServiceInstance.get_url()`](#client.ServiceInstance.get_url)
			* [`ServiceInstance.get_username()`](#client.ServiceInstance.get_username)
	+ [Set](#set)
		- [`Set`](#client.Set)
			* [`Set.default_project()`](#client.Set.default_project)
			* [`Set.default_space()`](#client.Set.default_space)
	+ [Shiny (IBM Cloud Pak for Data only)](#shiny-ibm-cloud-pak-for-data-only)
		- [`Shiny`](#client.Shiny)
			* [`Shiny.ConfigurationMetaNames`](#client.Shiny.ConfigurationMetaNames)
			* [`Shiny.create_revision()`](#client.Shiny.create_revision)
			* [`Shiny.delete()`](#client.Shiny.delete)
			* [`Shiny.download()`](#client.Shiny.download)
			* [`Shiny.get_details()`](#client.Shiny.get_details)
			* [`Shiny.get_href()`](#client.Shiny.get_href)
			* [`Shiny.get_id()`](#client.Shiny.get_id)
			* [`Shiny.get_revision_details()`](#client.Shiny.get_revision_details)
			* [`Shiny.get_uid()`](#client.Shiny.get_uid)
			* [`Shiny.list()`](#client.Shiny.list)
			* [`Shiny.list_revisions()`](#client.Shiny.list_revisions)
			* [`Shiny.store()`](#client.Shiny.store)
			* [`Shiny.update()`](#client.Shiny.update)
	+ [Software specifications](#software-specifications)
		- [`SwSpec`](#client.SwSpec)
			* [`SwSpec.ConfigurationMetaNames`](#client.SwSpec.ConfigurationMetaNames)
			* [`SwSpec.add_package_extension()`](#client.SwSpec.add_package_extension)
			* [`SwSpec.delete()`](#client.SwSpec.delete)
			* [`SwSpec.delete_package_extension()`](#client.SwSpec.delete_package_extension)
			* [`SwSpec.get_details()`](#client.SwSpec.get_details)
			* [`SwSpec.get_href()`](#client.SwSpec.get_href)
			* [`SwSpec.get_id()`](#client.SwSpec.get_id)
			* [`SwSpec.get_id_by_name()`](#client.SwSpec.get_id_by_name)
			* [`SwSpec.get_uid()`](#client.SwSpec.get_uid)
			* [`SwSpec.get_uid_by_name()`](#client.SwSpec.get_uid_by_name)
			* [`SwSpec.list()`](#client.SwSpec.list)
			* [`SwSpec.store()`](#client.SwSpec.store)
		- [`SwSpecMetaNames`](#metanames.SwSpecMetaNames)
	+ [Spaces](#spaces)
		- [`Spaces`](#client.Spaces)
			* [`Spaces.ConfigurationMetaNames`](#client.Spaces.ConfigurationMetaNames)
			* [`Spaces.MemberMetaNames`](#client.Spaces.MemberMetaNames)
			* [`Spaces.create_member()`](#client.Spaces.create_member)
			* [`Spaces.delete()`](#client.Spaces.delete)
			* [`Spaces.delete_member()`](#client.Spaces.delete_member)
			* [`Spaces.get_details()`](#client.Spaces.get_details)
			* [`Spaces.get_id()`](#client.Spaces.get_id)
			* [`Spaces.get_member_details()`](#client.Spaces.get_member_details)
			* [`Spaces.get_uid()`](#client.Spaces.get_uid)
			* [`Spaces.list()`](#client.Spaces.list)
			* [`Spaces.list_members()`](#client.Spaces.list_members)
			* [`Spaces.promote()`](#client.Spaces.promote)
			* [`Spaces.store()`](#client.Spaces.store)
			* [`Spaces.update()`](#client.Spaces.update)
			* [`Spaces.update_member()`](#client.Spaces.update_member)
		- [`SpacesMetaNames`](#metanames.SpacesMetaNames)
		- [`SpacesMemberMetaNames`](#metanames.SpacesMemberMetaNames)
	+ [Training](#training)
		- [`Training`](#client.Training)
			* [`Training.cancel()`](#client.Training.cancel)
			* [`Training.get_details()`](#client.Training.get_details)
			* [`Training.get_href()`](#client.Training.get_href)
			* [`Training.get_id()`](#client.Training.get_id)
			* [`Training.get_metrics()`](#client.Training.get_metrics)
			* [`Training.get_status()`](#client.Training.get_status)
			* [`Training.list()`](#client.Training.list)
			* [`Training.list_intermediate_models()`](#client.Training.list_intermediate_models)
			* [`Training.monitor_logs()`](#client.Training.monitor_logs)
			* [`Training.monitor_metrics()`](#client.Training.monitor_metrics)
			* [`Training.run()`](#client.Training.run)
		- [`TrainingConfigurationMetaNames`](#metanames.TrainingConfigurationMetaNames)
	+ [Enums](#module-ibm_watsonx_ai.utils.autoai.enums)
		- [`ClassificationAlgorithms`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms)
			* [`ClassificationAlgorithms.DT`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.DT)
			* [`ClassificationAlgorithms.EX_TREES`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.EX_TREES)
			* [`ClassificationAlgorithms.GB`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.GB)
			* [`ClassificationAlgorithms.LGBM`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.LGBM)
			* [`ClassificationAlgorithms.LR`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.LR)
			* [`ClassificationAlgorithms.RF`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.RF)
			* [`ClassificationAlgorithms.SnapBM`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.SnapBM)
			* [`ClassificationAlgorithms.SnapDT`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.SnapDT)
			* [`ClassificationAlgorithms.SnapLR`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.SnapLR)
			* [`ClassificationAlgorithms.SnapRF`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.SnapRF)
			* [`ClassificationAlgorithms.SnapSVM`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.SnapSVM)
			* [`ClassificationAlgorithms.XGB`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.XGB)
		- [`ClassificationAlgorithmsCP4D`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D)
			* [`ClassificationAlgorithmsCP4D.DT`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.DT)
			* [`ClassificationAlgorithmsCP4D.EX_TREES`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.EX_TREES)
			* [`ClassificationAlgorithmsCP4D.GB`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.GB)
			* [`ClassificationAlgorithmsCP4D.LGBM`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.LGBM)
			* [`ClassificationAlgorithmsCP4D.LR`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.LR)
			* [`ClassificationAlgorithmsCP4D.RF`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.RF)
			* [`ClassificationAlgorithmsCP4D.SnapBM`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.SnapBM)
			* [`ClassificationAlgorithmsCP4D.SnapDT`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.SnapDT)
			* [`ClassificationAlgorithmsCP4D.SnapLR`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.SnapLR)
			* [`ClassificationAlgorithmsCP4D.SnapRF`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.SnapRF)
			* [`ClassificationAlgorithmsCP4D.SnapSVM`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.SnapSVM)
			* [`ClassificationAlgorithmsCP4D.XGB`](#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.XGB)
		- [`DataConnectionTypes`](#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes)
			* [`DataConnectionTypes.CA`](#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes.CA)
			* [`DataConnectionTypes.CN`](#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes.CN)
			* [`DataConnectionTypes.DS`](#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes.DS)
			* [`DataConnectionTypes.FS`](#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes.FS)
			* [`DataConnectionTypes.S3`](#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes.S3)
		- [`Directions`](#ibm_watsonx_ai.utils.autoai.enums.Directions)
			* [`Directions.ASCENDING`](#ibm_watsonx_ai.utils.autoai.enums.Directions.ASCENDING)
			* [`Directions.DESCENDING`](#ibm_watsonx_ai.utils.autoai.enums.Directions.DESCENDING)
		- [`ForecastingAlgorithms`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms)
			* [`ForecastingAlgorithms.ARIMA`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.ARIMA)
			* [`ForecastingAlgorithms.BATS`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.BATS)
			* [`ForecastingAlgorithms.ENSEMBLER`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.ENSEMBLER)
			* [`ForecastingAlgorithms.HW`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.HW)
			* [`ForecastingAlgorithms.LR`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.LR)
			* [`ForecastingAlgorithms.RF`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.RF)
			* [`ForecastingAlgorithms.SVM`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.SVM)
		- [`ForecastingAlgorithmsCP4D`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D)
			* [`ForecastingAlgorithmsCP4D.ARIMA`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.ARIMA)
			* [`ForecastingAlgorithmsCP4D.BATS`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.BATS)
			* [`ForecastingAlgorithmsCP4D.ENSEMBLER`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.ENSEMBLER)
			* [`ForecastingAlgorithmsCP4D.HW`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.HW)
			* [`ForecastingAlgorithmsCP4D.LR`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.LR)
			* [`ForecastingAlgorithmsCP4D.RF`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.RF)
			* [`ForecastingAlgorithmsCP4D.SVM`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.SVM)
		- [`ForecastingPipelineTypes`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes)
			* [`ForecastingPipelineTypes.ARIMA`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMA)
			* [`ForecastingPipelineTypes.ARIMAX`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMAX)
			* [`ForecastingPipelineTypes.ARIMAX_DMLR`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMAX_DMLR)
			* [`ForecastingPipelineTypes.ARIMAX_PALR`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMAX_PALR)
			* [`ForecastingPipelineTypes.ARIMAX_RAR`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMAX_RAR)
			* [`ForecastingPipelineTypes.ARIMAX_RSAR`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMAX_RSAR)
			* [`ForecastingPipelineTypes.Bats`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.Bats)
			* [`ForecastingPipelineTypes.DifferenceFlattenEnsembler`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.DifferenceFlattenEnsembler)
			* [`ForecastingPipelineTypes.ExogenousDifferenceFlattenEnsembler`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousDifferenceFlattenEnsembler)
			* [`ForecastingPipelineTypes.ExogenousFlattenEnsembler`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousFlattenEnsembler)
			* [`ForecastingPipelineTypes.ExogenousLocalizedFlattenEnsembler`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousLocalizedFlattenEnsembler)
			* [`ForecastingPipelineTypes.ExogenousMT2RForecaster`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousMT2RForecaster)
			* [`ForecastingPipelineTypes.ExogenousRandomForestRegressor`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousRandomForestRegressor)
			* [`ForecastingPipelineTypes.ExogenousSVM`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousSVM)
			* [`ForecastingPipelineTypes.FlattenEnsembler`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.FlattenEnsembler)
			* [`ForecastingPipelineTypes.HoltWinterAdditive`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.HoltWinterAdditive)
			* [`ForecastingPipelineTypes.HoltWinterMultiplicative`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.HoltWinterMultiplicative)
			* [`ForecastingPipelineTypes.LocalizedFlattenEnsembler`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.LocalizedFlattenEnsembler)
			* [`ForecastingPipelineTypes.MT2RForecaster`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.MT2RForecaster)
			* [`ForecastingPipelineTypes.RandomForestRegressor`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.RandomForestRegressor)
			* [`ForecastingPipelineTypes.SVM`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.SVM)
			* [`ForecastingPipelineTypes.get_exogenous()`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.get_exogenous)
			* [`ForecastingPipelineTypes.get_non_exogenous()`](#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.get_non_exogenous)
		- [`ImputationStrategy`](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy)
			* [`ImputationStrategy.BEST_OF_DEFAULT_IMPUTERS`](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.BEST_OF_DEFAULT_IMPUTERS)
			* [`ImputationStrategy.CUBIC`](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.CUBIC)
			* [`ImputationStrategy.FLATTEN_ITERATIVE`](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.FLATTEN_ITERATIVE)
			* [`ImputationStrategy.LINEAR`](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.LINEAR)
			* [`ImputationStrategy.MEAN`](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.MEAN)
			* [`ImputationStrategy.MEDIAN`](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.MEDIAN)
			* [`ImputationStrategy.MOST_FREQUENT`](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.MOST_FREQUENT)
			* [`ImputationStrategy.NEXT`](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.NEXT)
			* [`ImputationStrategy.NO_IMPUTATION`](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.NO_IMPUTATION)
			* [`ImputationStrategy.PREVIOUS`](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.PREVIOUS)
			* [`ImputationStrategy.VALUE`](#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.VALUE)
		- [`Metrics`](#ibm_watsonx_ai.utils.autoai.enums.Metrics)
			* [`Metrics.ACCURACY_AND_DISPARATE_IMPACT_SCORE`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.ACCURACY_AND_DISPARATE_IMPACT_SCORE)
			* [`Metrics.ACCURACY_SCORE`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.ACCURACY_SCORE)
			* [`Metrics.AVERAGE_PRECISION_SCORE`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.AVERAGE_PRECISION_SCORE)
			* [`Metrics.EXPLAINED_VARIANCE_SCORE`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.EXPLAINED_VARIANCE_SCORE)
			* [`Metrics.F1_SCORE`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.F1_SCORE)
			* [`Metrics.F1_SCORE_MACRO`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.F1_SCORE_MACRO)
			* [`Metrics.F1_SCORE_MICRO`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.F1_SCORE_MICRO)
			* [`Metrics.F1_SCORE_WEIGHTED`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.F1_SCORE_WEIGHTED)
			* [`Metrics.LOG_LOSS`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.LOG_LOSS)
			* [`Metrics.MEAN_ABSOLUTE_ERROR`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.MEAN_ABSOLUTE_ERROR)
			* [`Metrics.MEAN_SQUARED_ERROR`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.MEAN_SQUARED_ERROR)
			* [`Metrics.MEAN_SQUARED_LOG_ERROR`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.MEAN_SQUARED_LOG_ERROR)
			* [`Metrics.MEDIAN_ABSOLUTE_ERROR`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.MEDIAN_ABSOLUTE_ERROR)
			* [`Metrics.PRECISION_SCORE`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.PRECISION_SCORE)
			* [`Metrics.PRECISION_SCORE_MACRO`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.PRECISION_SCORE_MACRO)
			* [`Metrics.PRECISION_SCORE_MICRO`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.PRECISION_SCORE_MICRO)
			* [`Metrics.PRECISION_SCORE_WEIGHTED`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.PRECISION_SCORE_WEIGHTED)
			* [`Metrics.R2_AND_DISPARATE_IMPACT_SCORE`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.R2_AND_DISPARATE_IMPACT_SCORE)
			* [`Metrics.R2_SCORE`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.R2_SCORE)
			* [`Metrics.RECALL_SCORE`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.RECALL_SCORE)
			* [`Metrics.RECALL_SCORE_MACRO`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.RECALL_SCORE_MACRO)
			* [`Metrics.RECALL_SCORE_MICRO`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.RECALL_SCORE_MICRO)
			* [`Metrics.RECALL_SCORE_WEIGHTED`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.RECALL_SCORE_WEIGHTED)
			* [`Metrics.ROC_AUC_SCORE`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.ROC_AUC_SCORE)
			* [`Metrics.ROOT_MEAN_SQUARED_ERROR`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.ROOT_MEAN_SQUARED_ERROR)
			* [`Metrics.ROOT_MEAN_SQUARED_LOG_ERROR`](#ibm_watsonx_ai.utils.autoai.enums.Metrics.ROOT_MEAN_SQUARED_LOG_ERROR)
		- [`MetricsToDirections`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections)
			* [`MetricsToDirections.ACCURACY`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.ACCURACY)
			* [`MetricsToDirections.AVERAGE_PRECISION`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.AVERAGE_PRECISION)
			* [`MetricsToDirections.EXPLAINED_VARIANCE`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.EXPLAINED_VARIANCE)
			* [`MetricsToDirections.F1`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.F1)
			* [`MetricsToDirections.F1_MACRO`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.F1_MACRO)
			* [`MetricsToDirections.F1_MICRO`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.F1_MICRO)
			* [`MetricsToDirections.F1_WEIGHTED`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.F1_WEIGHTED)
			* [`MetricsToDirections.NEG_LOG_LOSS`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_LOG_LOSS)
			* [`MetricsToDirections.NEG_MEAN_ABSOLUTE_ERROR`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_MEAN_ABSOLUTE_ERROR)
			* [`MetricsToDirections.NEG_MEAN_SQUARED_ERROR`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_MEAN_SQUARED_ERROR)
			* [`MetricsToDirections.NEG_MEAN_SQUARED_LOG_ERROR`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_MEAN_SQUARED_LOG_ERROR)
			* [`MetricsToDirections.NEG_MEDIAN_ABSOLUTE_ERROR`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_MEDIAN_ABSOLUTE_ERROR)
			* [`MetricsToDirections.NEG_ROOT_MEAN_SQUARED_ERROR`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_ROOT_MEAN_SQUARED_ERROR)
			* [`MetricsToDirections.NEG_ROOT_MEAN_SQUARED_LOG_ERROR`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_ROOT_MEAN_SQUARED_LOG_ERROR)
			* [`MetricsToDirections.NORMALIZED_GINI_COEFFICIENT`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NORMALIZED_GINI_COEFFICIENT)
			* [`MetricsToDirections.PRECISION`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.PRECISION)
			* [`MetricsToDirections.PRECISION_MACRO`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.PRECISION_MACRO)
			* [`MetricsToDirections.PRECISION_MICRO`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.PRECISION_MICRO)
			* [`MetricsToDirections.PRECISION_WEIGHTED`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.PRECISION_WEIGHTED)
			* [`MetricsToDirections.R2`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.R2)
			* [`MetricsToDirections.RECALL`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.RECALL)
			* [`MetricsToDirections.RECALL_MACRO`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.RECALL_MACRO)
			* [`MetricsToDirections.RECALL_MICRO`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.RECALL_MICRO)
			* [`MetricsToDirections.RECALL_WEIGHTED`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.RECALL_WEIGHTED)
			* [`MetricsToDirections.ROC_AUC`](#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.ROC_AUC)
		- [`PipelineTypes`](#ibm_watsonx_ai.utils.autoai.enums.PipelineTypes)
			* [`PipelineTypes.LALE`](#ibm_watsonx_ai.utils.autoai.enums.PipelineTypes.LALE)
			* [`PipelineTypes.SKLEARN`](#ibm_watsonx_ai.utils.autoai.enums.PipelineTypes.SKLEARN)
		- [`PositiveLabelClass`](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass)
			* [`PositiveLabelClass.AVERAGE_PRECISION_SCORE`](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.AVERAGE_PRECISION_SCORE)
			* [`PositiveLabelClass.F1_SCORE`](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.F1_SCORE)
			* [`PositiveLabelClass.F1_SCORE_MACRO`](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.F1_SCORE_MACRO)
			* [`PositiveLabelClass.F1_SCORE_MICRO`](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.F1_SCORE_MICRO)
			* [`PositiveLabelClass.F1_SCORE_WEIGHTED`](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.F1_SCORE_WEIGHTED)
			* [`PositiveLabelClass.PRECISION_SCORE`](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.PRECISION_SCORE)
			* [`PositiveLabelClass.PRECISION_SCORE_MACRO`](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.PRECISION_SCORE_MACRO)
			* [`PositiveLabelClass.PRECISION_SCORE_MICRO`](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.PRECISION_SCORE_MICRO)
			* [`PositiveLabelClass.PRECISION_SCORE_WEIGHTED`](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.PRECISION_SCORE_WEIGHTED)
			* [`PositiveLabelClass.RECALL_SCORE`](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.RECALL_SCORE)
			* [`PositiveLabelClass.RECALL_SCORE_MACRO`](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.RECALL_SCORE_MACRO)
			* [`PositiveLabelClass.RECALL_SCORE_MICRO`](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.RECALL_SCORE_MICRO)
			* [`PositiveLabelClass.RECALL_SCORE_WEIGHTED`](#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.RECALL_SCORE_WEIGHTED)
		- [`PredictionType`](#ibm_watsonx_ai.utils.autoai.enums.PredictionType)
			* [`PredictionType.BINARY`](#ibm_watsonx_ai.utils.autoai.enums.PredictionType.BINARY)
			* [`PredictionType.CLASSIFICATION`](#ibm_watsonx_ai.utils.autoai.enums.PredictionType.CLASSIFICATION)
			* [`PredictionType.FORECASTING`](#ibm_watsonx_ai.utils.autoai.enums.PredictionType.FORECASTING)
			* [`PredictionType.MULTICLASS`](#ibm_watsonx_ai.utils.autoai.enums.PredictionType.MULTICLASS)
			* [`PredictionType.REGRESSION`](#ibm_watsonx_ai.utils.autoai.enums.PredictionType.REGRESSION)
			* [`PredictionType.TIMESERIES_ANOMALY_PREDICTION`](#ibm_watsonx_ai.utils.autoai.enums.PredictionType.TIMESERIES_ANOMALY_PREDICTION)
		- [`RegressionAlgorithms`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms)
			* [`RegressionAlgorithms.DT`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.DT)
			* [`RegressionAlgorithms.EX_TREES`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.EX_TREES)
			* [`RegressionAlgorithms.GB`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.GB)
			* [`RegressionAlgorithms.LGBM`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.LGBM)
			* [`RegressionAlgorithms.LR`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.LR)
			* [`RegressionAlgorithms.RF`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.RF)
			* [`RegressionAlgorithms.RIDGE`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.RIDGE)
			* [`RegressionAlgorithms.SnapBM`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.SnapBM)
			* [`RegressionAlgorithms.SnapDT`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.SnapDT)
			* [`RegressionAlgorithms.SnapRF`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.SnapRF)
			* [`RegressionAlgorithms.XGB`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.XGB)
		- [`RegressionAlgorithmsCP4D`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D)
			* [`RegressionAlgorithmsCP4D.DT`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.DT)
			* [`RegressionAlgorithmsCP4D.EX_TREES`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.EX_TREES)
			* [`RegressionAlgorithmsCP4D.GB`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.GB)
			* [`RegressionAlgorithmsCP4D.LGBM`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.LGBM)
			* [`RegressionAlgorithmsCP4D.LR`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.LR)
			* [`RegressionAlgorithmsCP4D.RF`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.RF)
			* [`RegressionAlgorithmsCP4D.RIDGE`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.RIDGE)
			* [`RegressionAlgorithmsCP4D.SnapBM`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.SnapBM)
			* [`RegressionAlgorithmsCP4D.SnapDT`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.SnapDT)
			* [`RegressionAlgorithmsCP4D.SnapRF`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.SnapRF)
			* [`RegressionAlgorithmsCP4D.XGB`](#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.XGB)
		- [`RunStateTypes`](#ibm_watsonx_ai.utils.autoai.enums.RunStateTypes)
			* [`RunStateTypes.COMPLETED`](#ibm_watsonx_ai.utils.autoai.enums.RunStateTypes.COMPLETED)
			* [`RunStateTypes.FAILED`](#ibm_watsonx_ai.utils.autoai.enums.RunStateTypes.FAILED)
		- [`SamplingTypes`](#ibm_watsonx_ai.utils.autoai.enums.SamplingTypes)
			* [`SamplingTypes.FIRST_VALUES`](#ibm_watsonx_ai.utils.autoai.enums.SamplingTypes.FIRST_VALUES)
			* [`SamplingTypes.LAST_VALUES`](#ibm_watsonx_ai.utils.autoai.enums.SamplingTypes.LAST_VALUES)
			* [`SamplingTypes.RANDOM`](#ibm_watsonx_ai.utils.autoai.enums.SamplingTypes.RANDOM)
			* [`SamplingTypes.STRATIFIED`](#ibm_watsonx_ai.utils.autoai.enums.SamplingTypes.STRATIFIED)
		- [`TShirtSize`](#ibm_watsonx_ai.utils.autoai.enums.TShirtSize)
			* [`TShirtSize.L`](#ibm_watsonx_ai.utils.autoai.enums.TShirtSize.L)
			* [`TShirtSize.M`](#ibm_watsonx_ai.utils.autoai.enums.TShirtSize.M)
			* [`TShirtSize.S`](#ibm_watsonx_ai.utils.autoai.enums.TShirtSize.S)
			* [`TShirtSize.XL`](#ibm_watsonx_ai.utils.autoai.enums.TShirtSize.XL)
		- [`TimeseriesAnomalyPredictionAlgorithms`](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionAlgorithms)
			* [`TimeseriesAnomalyPredictionAlgorithms.Forecasting`](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionAlgorithms.Forecasting)
			* [`TimeseriesAnomalyPredictionAlgorithms.Relationship`](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionAlgorithms.Relationship)
			* [`TimeseriesAnomalyPredictionAlgorithms.Window`](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionAlgorithms.Window)
		- [`TimeseriesAnomalyPredictionPipelineTypes`](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes)
			* [`TimeseriesAnomalyPredictionPipelineTypes.PointwiseBoundedBATS`](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.PointwiseBoundedBATS)
			* [`TimeseriesAnomalyPredictionPipelineTypes.PointwiseBoundedBATSForceUpdate`](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.PointwiseBoundedBATSForceUpdate)
			* [`TimeseriesAnomalyPredictionPipelineTypes.PointwiseBoundedHoltWintersAdditive`](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.PointwiseBoundedHoltWintersAdditive)
			* [`TimeseriesAnomalyPredictionPipelineTypes.WindowLOF`](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.WindowLOF)
			* [`TimeseriesAnomalyPredictionPipelineTypes.WindowNN`](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.WindowNN)
			* [`TimeseriesAnomalyPredictionPipelineTypes.WindowPCA`](#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.WindowPCA)
		- [`Transformers`](#ibm_watsonx_ai.utils.autoai.enums.Transformers)
			* [`Transformers.ABS`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.ABS)
			* [`Transformers.CBRT`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.CBRT)
			* [`Transformers.COS`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.COS)
			* [`Transformers.CUBE`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.CUBE)
			* [`Transformers.DIFF`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.DIFF)
			* [`Transformers.DIVIDE`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.DIVIDE)
			* [`Transformers.FEATUREAGGLOMERATION`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.FEATUREAGGLOMERATION)
			* [`Transformers.ISOFORESTANOMALY`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.ISOFORESTANOMALY)
			* [`Transformers.LOG`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.LOG)
			* [`Transformers.MAX`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.MAX)
			* [`Transformers.MINMAXSCALER`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.MINMAXSCALER)
			* [`Transformers.NXOR`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.NXOR)
			* [`Transformers.PCA`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.PCA)
			* [`Transformers.PRODUCT`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.PRODUCT)
			* [`Transformers.ROUND`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.ROUND)
			* [`Transformers.SIGMOID`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.SIGMOID)
			* [`Transformers.SIN`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.SIN)
			* [`Transformers.SQRT`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.SQRT)
			* [`Transformers.SQUARE`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.SQUARE)
			* [`Transformers.STDSCALER`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.STDSCALER)
			* [`Transformers.SUM`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.SUM)
			* [`Transformers.TAN`](#ibm_watsonx_ai.utils.autoai.enums.Transformers.TAN)
		- [`VisualizationTypes`](#ibm_watsonx_ai.utils.autoai.enums.VisualizationTypes)
			* [`VisualizationTypes.INPLACE`](#ibm_watsonx_ai.utils.autoai.enums.VisualizationTypes.INPLACE)
			* [`VisualizationTypes.PDF`](#ibm_watsonx_ai.utils.autoai.enums.VisualizationTypes.PDF)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/fm_helpers.html








Helpers - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](#)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Helpers[¶](#helpers "Link to this heading")
===========================================




*class* ibm\_watsonx\_ai.foundation\_models\_manager.FoundationModelsManager(*client*)[[source]](_modules/ibm_watsonx_ai/foundation_models_manager.html#FoundationModelsManager)[¶](#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager "Link to this definition")


get\_custom\_model\_specs(*model\_id=None*, *limit=None*, *asynchronous=False*, *get\_all=False*)[[source]](_modules/ibm_watsonx_ai/foundation_models_manager.html#FoundationModelsManager.get_custom_model_specs)[¶](#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_custom_model_specs "Link to this definition")
Get details on available custom model(s) as dict or as generator (`asynchronous`).
If `asynchronous` or `get_all` is set, then `model_id` is ignored.



Parameters:
* **model\_id** (*str**,* *optional*) – Id of the model, defaults to None (all models specs are returned).
* **limit** (*int**,* *optional*) – limit number of fetched records
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks



Returns:
details of supported custom models, None if for given model\_id non is found



Return type:
dict or generator




**Example**



```
client.foundation_models.get_custom_models_spec()
client.foundation_models.get_custom_models_spec()
client.foundation_models.get_custom_models_spec(model_id='mistralai/Mistral-7B-Instruct-v0.2')
client.foundation_models.get_custom_models_spec(limit=20)
client.foundation_models.get_custom_models_spec(limit=20, get_all=True)
for spec in client.foundation_models.get_custom_model_specs(limit=20, asynchronous=True, get_all=True):
    print(spec, end="")

```





get\_embeddings\_model\_specs(*model\_id=None*, *limit=None*, *asynchronous=False*, *get\_all=False*)[[source]](_modules/ibm_watsonx_ai/foundation_models_manager.html#FoundationModelsManager.get_embeddings_model_specs)[¶](#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_embeddings_model_specs "Link to this definition")
Operation to retrieve the embeddings model specs.



Parameters:
* **model\_id** (*str**,* *optional*) – Id of the model, defaults to None (all models specs are returned).
* **limit** (*int**,* *optional*) – limit number of fetched records
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks



Returns:
embeddings model specs



Return type:
dict or generator




**Example**



```
client.foundation_models.get_embeddings_model_specs()
client.foundation_models.get_embeddings_model_specs('ibm/slate-125m-english-rtrvr')

```





get\_model\_lifecycle(*model\_id*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/foundation_models_manager.html#FoundationModelsManager.get_model_lifecycle)[¶](#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_model_lifecycle "Link to this definition")
Operation to retrieve the list of model lifecycle data.



Parameters:
**model\_id** (*str*) – the type of model to use



Returns:
list of deployed foundation model lifecycle data



Return type:
list




**Example**



```
client.foundation_models.get_model_lifecycle(
    model_id="ibm/granite-13b-instruct-v2"
    )

```





get\_model\_specs(*model\_id=None*, *limit=None*, *asynchronous=False*, *get\_all=False*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/foundation_models_manager.html#FoundationModelsManager.get_model_specs)[¶](#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_model_specs "Link to this definition")
Operations to retrieve the list of deployed foundation models specifications.



Parameters:
* **model\_id** (*str* *or* [*ModelTypes*](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes "ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes")*,* *optional*) – Id of the model, defaults to None (all models specs are returned).
* **limit** (*int**,* *optional*) – limit number of fetched records
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks



Returns:
list of deployed foundation model specs



Return type:
dict or generator




**Example**



```
# GET ALL MODEL SPECS
client.foundation_models.get_model_specs()

# GET MODEL SPECS BY MODEL_ID
client.foundation_models.get_model_specs(model_id="google/flan-ul2")

```





get\_model\_specs\_with\_prompt\_tuning\_support(*model\_id=None*, *limit=None*, *asynchronous=False*, *get\_all=False*)[[source]](_modules/ibm_watsonx_ai/foundation_models_manager.html#FoundationModelsManager.get_model_specs_with_prompt_tuning_support)[¶](#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_model_specs_with_prompt_tuning_support "Link to this definition")
Operations to query the details of the deployed foundation models with prompt tuning support.



Parameters:
* **model\_id** (*str**,* *optional*) – Id of the model, defaults to None (all models specs are returned).
* **limit** (*int**,* *optional*) – limit number of fetched records
* **asynchronous** (*bool**,* *optional*) – if True, it will work as a generator
* **get\_all** (*bool**,* *optional*) – if True, it will get all entries in ‘limited’ chunks



Returns:
list of deployed foundation model specs with prompt tuning support



Return type:
dict or generator




**Example**



```
client.foundation_models.get_model_specs_with_prompt_tuning_support()
client.foundation_models.get_model_specs_with_prompt_tuning_support('google/flan-t5-xl')

```






ibm\_watsonx\_ai.foundation\_models.get\_model\_specs(*url*, *model\_id=None*)[[source]](_modules/ibm_watsonx_ai/foundation_models/utils/utils.html#get_model_specs)[¶](#ibm_watsonx_ai.foundation_models.get_model_specs "Link to this definition")
Operations to retrieve the list of deployed foundation models specifications.


*Decrecated:* get\_model\_specs() function is deprecated from 1.0, please use client.foundation\_models.get\_model\_specs() function instead.



Parameters:
* **url** (*str*) – environment url
* **model\_id** (*Optional**[**str**,* [*ModelTypes*](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes "ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes")*]**,* *optional*) – Id of the model, defaults to None (all models specs are returned).



Returns:
list of deployed foundation model specs



Return type:
dict




**Example**



```
from ibm_watsonx_ai.foundation_models import get_model_specs

# GET ALL MODEL SPECS
get_model_specs(
    url="https://us-south.ml.cloud.ibm.com"
    )

# GET MODEL SPECS BY MODEL_ID
get_model_specs(
    url="https://us-south.ml.cloud.ibm.com",
    model_id="google/flan-ul2"
    )

```





ibm\_watsonx\_ai.foundation\_models.get\_model\_lifecycle(*url*, *model\_id*)[[source]](_modules/ibm_watsonx_ai/foundation_models/utils/utils.html#get_model_lifecycle)[¶](#ibm_watsonx_ai.foundation_models.get_model_lifecycle "Link to this definition")
Operation to retrieve the list of model lifecycle data.


*Decrecated:* get\_model\_lifecycle() function is deprecated from 1.0, please use client.foundation\_models.get\_model\_lifecycle() function instead.



Parameters:
* **url** (*str*) – environment url
* **model\_id** (*str*) – the type of model to use



Returns:
list of deployed foundation model lifecycle data



Return type:
list




**Example**



```
from ibm_watsonx_ai.foundation_models import get_model_lifecycle

get_model_lifecycle(
    url="https://us-south.ml.cloud.ibm.com",
    model_id="ibm/granite-13b-instruct-v2"
    )

```





ibm\_watsonx\_ai.foundation\_models.get\_model\_specs\_with\_prompt\_tuning\_support(*url*)[[source]](_modules/ibm_watsonx_ai/foundation_models/utils/utils.html#get_model_specs_with_prompt_tuning_support)[¶](#ibm_watsonx_ai.foundation_models.get_model_specs_with_prompt_tuning_support "Link to this definition")
Operations to query the details of the deployed foundation models with prompt tuning support.


*Decrecated:* get\_model\_specs\_with\_prompt\_tuning\_support() function is deprecated from 1.0, please use client.foundation\_models.get\_model\_specs\_with\_prompt\_tuning\_support() function instead.



Parameters:
**url** (*str*) – environment url



Returns:
list of deployed foundation model specs with prompt tuning support



Return type:
dict




**Example**



```
from ibm_watsonx_ai.foundation_models import get_model_specs_with_prompt_tuning_support

get_model_specs_with_prompt_tuning_support(
    url="https://us-south.ml.cloud.ibm.com"
    )

```





ibm\_watsonx\_ai.foundation\_models.get\_supported\_tasks(*url*)[[source]](_modules/ibm_watsonx_ai/foundation_models/utils/utils.html#get_supported_tasks)[¶](#ibm_watsonx_ai.foundation_models.get_supported_tasks "Link to this definition")
Operation to retrieve the list of tasks that are supported by the foundation models.



Parameters:
**url** (*str*) – environment url



Returns:
list of tasks that are supported by the foundation models



Return type:
dict




**Example**



```
from ibm_watsonx_ai.foundation_models import get_supported_tasks

get_supported_tasks(
    url="https://us-south.ml.cloud.ibm.com"
    )

```








[Next

Custom models](fm_working_with_custom_models.html)
[Previous

Extensions](fm_extensions.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Helpers](#)
	+ [`FoundationModelsManager`](#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager)
		- [`FoundationModelsManager.get_custom_model_specs()`](#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_custom_model_specs)
		- [`FoundationModelsManager.get_embeddings_model_specs()`](#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_embeddings_model_specs)
		- [`FoundationModelsManager.get_model_lifecycle()`](#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_model_lifecycle)
		- [`FoundationModelsManager.get_model_specs()`](#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_model_specs)
		- [`FoundationModelsManager.get_model_specs_with_prompt_tuning_support()`](#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_model_specs_with_prompt_tuning_support)
	+ [`get_model_specs()`](#ibm_watsonx_ai.foundation_models.get_model_specs)
	+ [`get_model_lifecycle()`](#ibm_watsonx_ai.foundation_models.get_model_lifecycle)
	+ [`get_model_specs_with_prompt_tuning_support()`](#ibm_watsonx_ai.foundation_models.get_model_specs_with_prompt_tuning_support)
	+ [`get_supported_tasks()`](#ibm_watsonx_ai.foundation_models.get_supported_tasks)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/base.html








Base - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](#)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Base[¶](#base "Link to this heading")
=====================================



APIClient[¶](#apiclient "Link to this heading")
-----------------------------------------------




*class* client.APIClient(*credentials=None*, *project\_id=None*, *space\_id=None*, *verify=None*, *\*\*kwargs*)[[source]](_modules/client.html#APIClient)[¶](#client.APIClient "Link to this definition")
The main class of ibm\_watsonx\_ai. The very heart of the module. APIClient contains objects that manage the service reasources.



To explore how to use APIClient, refer to:* [Setup](setup.html#setup) - to check correct initialization of APIClient for a specific environment.
* [Core](core_api.html#core) - to explore core properties of an APIClient object.





Parameters:
* **url** (*str*) – URL of the service
* **credentials** ([*Credentials*](#credentials.Credentials "credentials.Credentials")) – credentials used to connect with the service
* **project\_id** (*str**,* *optional*) – ID of the project that is used
* **space\_id** (*str**,* *optional*) – ID of deployment space that is used
* **verify** (*bool**,* *optional*) – certificate verification flag, deprecated, use Credentials(verify=…) to set verify




**Example**



```
from ibm_watsonx_ai import APIClient, Credentials

credentials = Credentials(
    url = "<url>",
    api_key = "<api_key>"
)

client = APIClient(credentials, space_id="<space_id>")

client.models.list()
client.deployments.get_details()

client.set.default_project("<project_id>")

...

```




set\_headers(*headers*)[[source]](_modules/client.html#APIClient.set_headers)[¶](#client.APIClient.set_headers "Link to this definition")
Method which allows refresh/set new User Request Headers.



Parameters:
**headers** (*dict*) – User Request Headers




**Examples**



```
headers = {
    'Authorization': 'Bearer <USER AUTHORIZATION TOKEN>',
    'User-Agent': 'ibm-watsonx-ai/1.0.1 (lang=python; arch=x86_64; os=darwin; python.version=3.10.13)',
    'X-Watson-Project-ID': '<PROJECT ID>',
    'Content-Type': 'application/json'
}

client.set_headers(headers)

```





set\_token(*token*)[[source]](_modules/client.html#APIClient.set_token)[¶](#client.APIClient.set_token "Link to this definition")
Method which allows refresh/set new User Authorization Token.



Parameters:
**token** (*str*) – User Authorization Token




**Examples**



```
client.set_token("<USER AUTHORIZATION TOKEN>")

```






Credentials[¶](#credentials "Link to this heading")
---------------------------------------------------




*class* credentials.Credentials(*\**, *url=None*, *api\_key=None*, *name=None*, *iam\_serviceid\_crn=None*, *token=None*, *projects\_token=None*, *username=None*, *password=None*, *instance\_id=None*, *version=None*, *bedrock\_url=None*, *proxies=None*, *verify=None*)[[source]](_modules/credentials.html#Credentials)[¶](#credentials.Credentials "Link to this definition")
This class encapsulate passed credentials and additional params.



Parameters:
* **url** (*str*) – URL of the service
* **api\_key** (*str**,* *optional*) – service API key used in API key authentication
* **name** (*str**,* *optional*) – service name used during space creation for a Cloud environment
* **iam\_serviceid\_crn** (*str**,* *optional*) – service CRN used during space creation for a Cloud environment
* **token** (*str**,* *optional*) – service token, used in token authentication
* **projects\_token** (*str**,* *optional*) – service projects token used in token authentication
* **username** (*str**,* *optional*) – username, used in username/password or username/api\_key authentication, applicable for ICP only
* **password** (*str**,* *optional*) – password, used in username/password authentication, applicable for ICP only
* **instance\_id** (*str**,* *optional*) – instance ID, mandatory for ICP
* **version** (*str**,* *optional*) – ICP version, mandatory for ICP
* **bedrock\_url** (*str**,* *optional*) – Bedrock URL, applicable for ICP only
* **proxies** (*dict**,* *optional*) – dictionary of proxies, containing protocol and URL mapping (example: { “https”: “https://example.url.com” })
* **verify** (*bool**,* *optional*) – certificate verification flag






*static* from\_dict(*credentials*, *\_verify=None*)[[source]](_modules/credentials.html#Credentials.from_dict)[¶](#credentials.Credentials.from_dict "Link to this definition")
Create a Credentials object from dictionary.



Parameters:
**credentials** (*dict*) – credentials in the dictionary



Returns:
initialised credentials object



Return type:
[Credentials](#credentials.Credentials "credentials.Credentials")




**Example**



```
from ibm_watsonx_ai import Credentials

credentials = Credentials.from_dict({
    'url': "<url>",
    'apikey': "<api_key>"
})

```





to\_dict()[[source]](_modules/credentials.html#Credentials.to_dict)[¶](#credentials.Credentials.to_dict "Link to this definition")
Get dictionary from the Credentials object.



Returns:
dictionary with credentials



Return type:
dict




**Example**



```
from ibm_watsonx_ai import Credentials

credentials = Credentials.from_dict({
    'url': "<url>",
    'apikey': "<api_key>"
})

credentials_dict = credentials.to_dict()

```










[Next

Core](core_api.html)
[Previous

API](api.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Base](#)
	+ [APIClient](#apiclient)
		- [`APIClient`](#client.APIClient)
			* [`APIClient.set_headers()`](#client.APIClient.set_headers)
			* [`APIClient.set_token()`](#client.APIClient.set_token)
	+ [Credentials](#credentials)
		- [`Credentials`](#credentials.Credentials)
			* [`Credentials.from_dict()`](#credentials.Credentials.from_dict)
			* [`Credentials.to_dict()`](#credentials.Credentials.to_dict)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/migration_v1.html








V1 Migration Guide - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](#)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





V1 Migration Guide[¶](#v1-migration-guide "Link to this heading")
=================================================================


New `ibm_watsonx_ai` Python SDK 1.0 is backward compatible in most common scenarios,
but few minor adjustments in code may be needed.



What’s new?[¶](#what-s-new "Link to this heading")
--------------------------------------------------



### Refactor and Cleanup[¶](#refactor-and-cleanup "Link to this heading")


This major release marks a thorough refactoring and cleanup of the codebase.
We’ve diligently worked to improve code readability, and enhance overall performance.
As a result, users can expect a more efficient and maintainable package.




### New authorization[¶](#new-authorization "Link to this heading")


In `ibm_watsonx_ai` 1.0 new class `Credentials` was introduced which replaces the dictionary with credentials.
See examples and more in docs: [Authentication Cloud](setup_cloud.html) and [Authentication](setup_cpd.html)


✅ New approach



```
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                   token = "***********",
                  )

client = APIClient(credentials)

```


❌ Old approach



```
from ibm_watsonx_ai import APIClient

credentials = {
                   "url": "https://us-south.ml.cloud.ibm.com",
                   "token":"***********",
                  }

client = APIClient(credentials)

```




### Easier project and space setup[¶](#easier-project-and-space-setup "Link to this heading")


Starting from `ibm_watsonx_ai` 1.0 release users can set the default space
or project while initializing APIClient object.


✅ New approach



```
client_project = APIClient(credentials, project_id = project_id)
client_space = APIClient(credentials, space_id = space_id)

```


❌ Old approach



```
client_project = APIClient(credentials)
client_project.set.default_project(project_id)

client_space = APIClient(credentials)
client_space.set.default_space(space_id)

```




### Foundation models functions moved under APIClient[¶](#foundation-models-functions-moved-under-apiclient "Link to this heading")


Starting from `ibm_watsonx_ai` 1.0 foundation models functions (`get_model_specs`, `get_custom_model_specs`,
`get_model_lifecycle`, `get_model_specs_with_prompt_tuning_support`, `get_embedding_model_specs`) are moved into `client.foundation_models` object.


Old versions of the functions are not removed, however, as they lack authentication, they will no longer work.


✅ New approach - Using functions from client.foundation\_models.



```
client.foundation_models.get_model_specs()

client.foundation_models.get_custom_model_specs()
client.foundation_models.get_custom_model_specs(
    model_id='mistralai/Mistral-7B-Instruct-v0.2'
)
client.foundation_models.get_custom_model_specs(limit=20)
client.foundation_models.get_custom_model_specs(limit=20, get_all=True)

client.foundation_models.get_model_lifecycle(
    model_id="ibm/granite-13b-instruct-v2"
)

client.foundation_models.get_model_specs_with_prompt_tuning_support()

client.foundation_models.get_embeddings_model_specs()

```


❌ Old approach - Using functions from ibm\_watsonx\_ai.foundation\_models module.



```
from ibm_watsonx_ai.foundation_models import (
    get_model_specs,
    get_custom_models_spec,
    get_model_lifecycle,
    get_model_specs_with_prompt_tuning_support,
    get_embedding_model_specs
)

get_model_specs(
    url="https://us-south.ml.cloud.ibm.com"
)

get_custom_models_spec(api_client=client)
get_custom_models_spec(credentials=credentials)
get_custom_models_spec(api_client=client, model_id='mistralai/Mistral-7B-Instruct-v0.2')
get_custom_models_spec(api_client=client, limit=20)
get_custom_models_spec(api_client=client, limit=20, get_all=True)

get_model_lifecycle(
    url="https://us-south.ml.cloud.ibm.com",
    model_id="ibm/granite-13b-instruct-v2"
)

get_model_specs_with_prompt_tuning_support(
    url="https://us-south.ml.cloud.ibm.com"
)

get_embedding_model_specs(
    url="https://us-south.ml.cloud.ibm.com"
)

```




### Breaking changes[¶](#breaking-changes "Link to this heading")


* `client.<resource>.list()` methods don’t print the table with listed assets. They return the table as `pandas.DataFrame`. The optional parameter for the methods `return_as_df` was removed.


✅ New approach - Access to credentials fields as class attributes.



```
conn_list = client.connections.list()  ## table not printed

```


❌ Old approach - Table with listed resources printed.



```
conn_list = client.connections.list()
### Table returned as pandas.DataFrame `conn_list` object and printed output:
---------------------------  ------------------------------------  --------------------  ------------------------------------
NAME                         ID                                    CREATED               DATASOURCE_TYPE_ID
Connection to COS            71738a79-6585-4f33-bf4a-18907abcf06a  2024-04-25T10:42:23Z  193a97c1-4475-4a19-b90c-295c4fdc6517
---------------------------  ------------------------------------  --------------------  ------------------------------------

```


* Methods and parameters that were marked as deprecated in `ibm_watsonx_ai v0` were removed. For example:


✅ New approach - Use method that replaced deprecated one, for `get_uid` it’s `get_id`



```
asset_id = client.data_assets.get_id(asset_details)

```


❌ Old approach - Deprecated `get_uid` method called



```
client.data_assets.get_uid(asset_details)

```


* `client.credentials` and `client.wml_credentials` returns `Credentials` object instead of dictionary.


✅ New approach - Access to credentials fields as class attributes.



```
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                   token = "***********",
                  )

client = APIClient(credentials)

url = client.credentials.url
# OR
url = credentials.url

```


❌ Old approach - Access to credentials fields as keys in dictionary.



```
from ibm_watsonx_ai import APIClient

credentials = {
                   "url": "https://us-south.ml.cloud.ibm.com",
                   "token":"***********",
                  }
client = APIClient(credentials)
url = client.wml_credentials.get('url')

```


* Parameter changes in `service.score(...)` for online deployment


`service.score(...)` has additional `forecast_window` parameter added before `transaction_id` parameter.
Additionally, `service.score(...)` function will require from 1.0.0 to pass all named parameters except `payload` parameter.


✅ New approach - named parameters and new parameter.



```
predictions = service.score({
        "observations": AbstractTestTSAsync.observations,
        "supporting_features": AbstractTestTSAsync.supporting_features,
    },
    forecast_window=1,
    transaction_id=transaction_id,
)

```


❌ Old approach - all parameters not named, `forecast_window` parameter absent



```
predictions = service.score({
        "observations": AbstractTestTSAsync.observations,
        "supporting_features": AbstractTestTSAsync.supporting_features,
    },
    transaction_id,
)

```




### Deprecations[¶](#deprecations "Link to this heading")


* Initializing `APIClient` with `credentials` as dictionary is deprecated.
* All parameters with `wml` in name are deprecated. They were renamed to:
	+ `wml_credentials` -> `credentials`
	+ `wml_client` -> `api_client`.
* `id` naming was aligned in parameters and methods, now all `uid` should be replaced by `id`. Methods that were using `uid` were either removed or mark as deprecated.









[Next

Changelog](changelog.html)
[Previous

Migration from `ibm_watson_machine_learning`](migration.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [V1 Migration Guide](#)
	+ [What’s new?](#what-s-new)
		- [Refactor and Cleanup](#refactor-and-cleanup)
		- [New authorization](#new-authorization)
		- [Easier project and space setup](#easier-project-and-space-setup)
		- [Foundation models functions moved under APIClient](#foundation-models-functions-moved-under-apiclient)
		- [Breaking changes](#breaking-changes)
		- [Deprecations](#deprecations)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/setup_cloud.html








IBM watsonx.ai for IBM Cloud - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](#)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





IBM watsonx.ai for IBM Cloud[¶](#ibm-watsonx-ai-for-ibm-cloud "Link to this heading")
=====================================================================================



Requirements[¶](#requirements "Link to this heading")
-----------------------------------------------------


For information on how to start working with IBM watsonx.ai for IBM Cloud, refer to [Getting started with Cloud Pak for Data as a Service](https://dataplatform.cloud.ibm.com/docs/content/wsj/getting-started/get-started-wdp.html?audience=wdp&context=wx).




Supported machine learning frameworks[¶](#supported-machine-learning-frameworks "Link to this heading")
-------------------------------------------------------------------------------------------------------


For a list of supported machine learning frameworks (models) on IBM watsonx.ai for IBM Cloud, refer to [Supported frameworks and software specifications](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/pm_service_supported_frameworks.html?context=wx&audience=wdp&locale=en).




Authentication[¶](#authentication "Link to this heading")
---------------------------------------------------------


To use watsonx.ai APIs, create an instance of APIClient with authentication details.


**Note:** Depending on the region of your provisioned service instance, use one of the following as your URL:



> * Dallas: https://us-south.ml.cloud.ibm.com
> * London: https://eu-gb.ml.cloud.ibm.com
> * Frankfurt: https://eu-de.ml.cloud.ibm.com
> * Tokyo: https://jp-tok.ml.cloud.ibm.com


**Note:** To determine your api\_key, refer to [IBM Cloud console API keys](https://cloud.ibm.com/iam/apikeys).


Example of creating the client using an API key:



```
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                   api_key = "***********"
                  )

client = APIClient(credentials)

```


Example of creating the client using a token:



```
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                   token = "***********",
                  )

client = APIClient(credentials)

```


**Note:** Setting a default space ID or project ID is mandatory. For details, refer to the `client.set.default_space()` API in this document.



Hint


You can copy the project\_id from the Project’s Manage tab (Project -> Manage -> General -> Details).





Firewall settings[¶](#firewall-settings "Link to this heading")
---------------------------------------------------------------


Although the above setup is sufficient for most environments, environments behind a firewall may need an additional adjustment.
The following endpoints are used by `ibm-watsonx-ai` and need to be whitelisted to ensure correct functioning of the module:



```
https://jp-tok.ml.cloud.ibm.com
https://eu-gb.ml.cloud.ibm.com
https://eu-de.ml.cloud.ibm.com
https://us-south.ml.cloud.ibm.com

https://api.jp-tok.dataplatform.cloud.ibm.com
https://api.eu-gb.dataplatform.cloud.ibm.com
https://api.eu-de.dataplatform.cloud.ibm.com
https://api.dataplatform.cloud.ibm.com
https://api.jp-tok.dataplatform.cloud.ibm.com

https://iam.cloud.ibm.com

```








[Next

IBM watsonx.ai software](setup_cpd.html)
[Previous

Setup](setup.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [IBM watsonx.ai for IBM Cloud](#)
	+ [Requirements](#requirements)
	+ [Supported machine learning frameworks](#supported-machine-learning-frameworks)
	+ [Authentication](#authentication)
	+ [Firewall settings](#firewall-settings)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/autoai_working_with_class_and_optimizer.html








Working with AutoAI class and optimizer - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](#)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Working with AutoAI class and optimizer[¶](#working-with-autoai-class-and-optimizer "Link to this heading")
===========================================================================================================


The [AutoAI experiment class](autoai_experiment.html#autoai-class) is responsible for creating experiments and scheduling training.
All experiment results are stored automatically in the user-specified Cloud Object Storage (COS).
Then the AutoAI feature can fetch the results and provide them directly to the user for further usage.



Configure optimizer with one data source[¶](#configure-optimizer-with-one-data-source "Link to this heading")
-------------------------------------------------------------------------------------------------------------


For an AutoAI object initialization, you need watsonx.ai credentials (with your API key and URL) and either the `project_id` or `space_id`.



Hint


You can copy the project\_id from the Project’s Manage tab (Project -> Manage -> General -> Details).




```
from ibm_watsonx_ai.experiment import AutoAI

experiment = AutoAI(wx_credentials,
    space_id='76g53e0-0b32-4a0e-9152-3d50324855ddb'
)

pipeline_optimizer = experiment.optimizer(
            name='test name',
            desc='test description',
            prediction_type=AutoAI.PredictionType.BINARY,
            prediction_column='y',
            scoring=AutoAI.Metrics.ACCURACY_SCORE,
            test_size=0.1,
            max_num_daub_ensembles=1,
            train_sample_rows_test_size=1.,
            daub_include_only_estimators = [
                 AutoAI.ClassificationAlgorithms.XGB,
                 AutoAI.ClassificationAlgorithms.LGBM
                 ],
            cognito_transform_names = [
                 AutoAI.Transformers.SUM,
                 AutoAI.Transformers.MAX
                 ]
        )

```




Configure optimizer for time series forecasting[¶](#configure-optimizer-for-time-series-forecasting "Link to this heading")
---------------------------------------------------------------------------------------------------------------------------


**Note:** Supported for IBM Cloud Pak for Data 4.0 and up.


Time series forecasting is a special AutoAI prediction scenario with specific parameters used to configure forecasting. These parameters include:
`prediction_columns`, `timestamp_column_name`, `backtest_num`, `lookback_window`, `forecast_window`, and `backtest_gap_length`.



```
from ibm_watsonx_ai.experiment import AutoAI

experiment = AutoAI(wx_credentials,
    space_id='76g53e0-0b32-4a0e-9152-3d50324855ddb')
)

pipeline_optimizer = experiment.optimizer(
    name='forecasting optimiser',
    desc='description',
    prediction_type=experiment.PredictionType.FORECASTING,
    prediction_columns=['value'],
    timestamp_column_name='timestamp',
    backtest_num=4,
    lookback_window=5,
    forecast_window=2,
    holdout_size=0.05,
    max_number_of_estimators=1,
    include_only_estimators=[AutoAI.ForecastingAlgorithms.ENSEMBLER],
    t_shirt_size=TShirtSize.L
)

```


Optimizer and deployment fitting procedures are the same as in the example scenario above.




Configure optimizer for time series forecasting with supporting features[¶](#configure-optimizer-for-time-series-forecasting-with-supporting-features "Link to this heading")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


**Note** Supported for IBM Cloud and IBM Cloud Pak for Data version 4.5 and up.


Additional parameters can be passed to run time series forecasting scenarios with supporting features:
`feature_columns`, `pipeline_types`, `supporting_features_at_forecast`


For more information about supporting features, refer to [time series documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/autoai-timeseries.html?context=cpdaas&audience=wdp).



```
from ibm_watsonx_ai.experiment import AutoAI
from ibm_watsonx_ai.utils.autoai.enums import ForecastingPipelineTypes

experiment = AutoAI(wx_credentials,
    space_id='76g53e0-0b32-4a0e-9152-3d50324855ddb')
)

pipeline_optimizer = experiment.optimizer(
    name='forecasting optimizer',
    desc='description',
    prediction_type=experiment.PredictionType.FORECASTING,
    prediction_columns=['value'],
    timestamp_column_name='week',
    feature_columns=['a', 'b', 'value'],
    pipeline_types=[ForecastingPipelineTypes.FlattenEnsembler] + ForecastingPipelineTypes.get_exogenous(),
    supporting_features_at_forecast=True
)

```


Predicting for time series forecasting scenario with supporting features:



```
# Example data:
#   new_observations:
#       week       a   b  value
#       14.0       0   0  134
#       15.0       1   4  96
#       ...
#
#   supporting_features:
#       week       a   b
#       16.0       1   3
#       ...

# with DataFrame or np.array:
pipeline_optimizer.predict(new_observations, supporting_features=supporting_features)

```


Online scoring for time series forecasting scenario with supporting features:



```
# with DataFrame:
web_service.score(payload={'observations': new_observations_df, 'supporting_features': supporting_features_df})

```


Batch scoring for time series forecasting scenario with supporting features:



```
# with DataFrame:
batch_service.run_job(payload={'observations': new_observations_df, 'supporting_features': supporting_features_df})

# with DataConnection:
batch_service.run_job(payload={'observations': new_observations_data_connection, 'supporting_features': supporting_features_data_connection})

```




Get configuration parameters[¶](#get-configuration-parameters "Link to this heading")
-------------------------------------------------------------------------------------


To see the current configuration parameters, call the `get_params()` method.



```
config_parameters = pipeline_optimizer.get_params()
print(config_parameters)
{
    'name': 'test name',
    'desc': 'test description',
    'prediction_type': 'classification',
    'prediction_column': 'y',
    'scoring': 'roc_auc',
    'test_size': 0.1,
    'max_num_daub_ensembles': 1
}

```




Fit optimizer[¶](#fit-optimizer "Link to this heading")
-------------------------------------------------------


To schedule an AutoAI experiment, call the `fit()` method. This will trigger a training and an optimization process on watsonx.ai. The `fit()` method can be synchronous (`background_mode=False`) or asynchronous (`background_mode=True`).
If you don’t want to wait for the fit to end, invoke the async version. It immediately returns only fit/run details.
If you invoke the async version, you see a progress bar with information about the learning/optimization process.



```
fit_details = pipeline_optimizer.fit(
        training_data_references=[training_data_connection],
        training_results_reference=results_connection,
        background_mode=True)

# OR

fit_details = pipeline_optimizer.fit(
        training_data_references=[training_data_connection],
        training_results_reference=results_connection,
        background_mode=False)

```


To run an AutoAI experiment with separate holdout data you can use the `fit()` method with the `test_data_references` parameter. See the example below:



```
fit_details = pipeline_optimizer.fit(
        training_data_references=[training_data_connection],
        test_data_references=[test_data_connection],
        training_results_reference=results_connection)

```




Get the run status and run details[¶](#get-the-run-status-and-run-details "Link to this heading")
-------------------------------------------------------------------------------------------------


If you use the `fit()` method asynchronously, you can monitor the run/fit details and status using the following two methods:



```
status = pipeline_optimizer.get_run_status()
print(status)
'running'

# OR

'completed'

run_details = pipeline_optimizer.get_run_details()
print(run_details)
{'entity': {'pipeline': {'href': '/v4/pipelines/5bfeb4c5-90df-48b8-9e03-ba232d8c0838'},
        'results_reference': {'connection': { 'id': ...},
                              'location': {'bucket': '...',
                                           'logs': '53c8cb7b-c8b5-44aa-8b52-6fde3c588462',
                                           'model': '53c8cb7b-c8b5-44aa-8b52-6fde3c588462/model',
                                           'path': '.',
                                           'pipeline': './33825fa2-5fca-471a-ab1a-c84820b3e34e/pipeline.json',
                                           'training': './33825fa2-5fca-471a-ab1a-c84820b3e34e',
                                           'training_status': './33825fa2-5fca-471a-ab1a-c84820b3e34e/training-status.json'},
                              'type': 'connected_asset'},
        'space': {'href': '/v4/spaces/71ab11ea-bb77-4ae6-b98a-a77f30ade09d'},
        'status': {'completed_at': '2020-02-17T10:46:32.962Z',
                   'message': {'level': 'info',
                               'text': 'Training job '
                                       '33825fa2-5fca-471a-ab1a-c84820b3e34e '
                                       'completed'},
                   'state': 'completed'},
        'training_data_references': [{'connection': {'id': '...'},
                                      'location': {'bucket': '...',
                                                   'path': '...'},
                                      'type': 'connected_asset'}]},
 'metadata': {'created_at': '2020-02-17T10:44:22.532Z',
              'guid': '33825fa2-5fca-471a-ab1a-c84820b3e34e',
              'href': '/v4/trainings/33825fa2-5fca-471a-ab1a-c84820b3e34e',
              'id': '33825fa2-5fca-471a-ab1a-c84820b3e34e',
              'modified_at': '2020-02-17T10:46:32.987Z'}}

```




Get data connections[¶](#get-data-connections "Link to this heading")
---------------------------------------------------------------------


The `data_connections` list contains all the training connections that you referenced while calling the `fit()` method.



```
data_connections = pipeline_optimizer.get_data_connections()

```




Pipeline summary[¶](#pipeline-summary "Link to this heading")
-------------------------------------------------------------


It is possible to get a ranking of all the computed pipeline models, sorted based on a scoring metric supplied when configuring the optimizer (`scoring` parameter). The output type is a `pandas.DataFrame` with pipeline names, computation timestamps, machine learning metrics, and the number of enhancements implemented in each of the pipelines.



```
results = pipeline_optimizer.summary()
print(results)
               Number of enhancements  ...  training_f1
Pipeline Name                          ...
Pipeline_4                          3  ...     0.555556
Pipeline_3                          2  ...     0.554978
Pipeline_2                          1  ...     0.503175
Pipeline_1                          0  ...     0.529928

```




Get pipeline details[¶](#get-pipeline-details "Link to this heading")
---------------------------------------------------------------------


To see the pipeline composition’s steps and nodes, use the `get_pipeline_details()` method.
If you leave `pipeline_name` empty, the method returns the details of the best computed pipeline.



```
pipeline_params = pipeline_optimizer.get_pipeline_details(pipeline_name='Pipeline_1')
print(pipeline_params)
{
    'composition_steps': [
        'TrainingDataset_full_199_16', 'Split_TrainingHoldout',
        'TrainingDataset_full_179_16', 'Preprocessor_default', 'DAUB'
        ],
    'pipeline_nodes': [
        'PreprocessingTransformer', 'LogisticRegressionEstimator'
        ]
}

```




Get pipeline[¶](#get-pipeline "Link to this heading")
-----------------------------------------------------


Use the `get_pipeline()` method to load a specific pipeline. By default, `get_pipeline()` returns a Lale pipeline. For information on Lale pipelines, refer to the [Lale library](https://github.com/ibm/lale).



```
pipeline = pipeline_optimizer.get_pipeline(pipeline_name='Pipeline_4')
print(type(pipeline))
'lale.operators.TrainablePipeline'

```


You can also load a pipeline as a scikit learn (sklearn) pipeline model type.



```
pipeline = pipeline_optimizer.get_pipeline(pipeline_name='Pipeline_4', astype=AutoAI.PipelineTypes.SKLEARN)
print(type(pipeline))
# <class 'sklearn.pipeline.Pipeline'>

```




Working with deployments[¶](#working-with-deployments "Link to this heading")
-----------------------------------------------------------------------------


This section describes classes that enable you to work with watsonx.ai deployments.




Web Service[¶](#web-service "Link to this heading")
---------------------------------------------------


Web Service is an online type of deployment. With Web Service, you can upload and deploy your model to score it through an online web service. You must pass the location where the training was performed using `source_space_id` or `source_project_id`. You can deploy the model to any space or project by providing the `target_space_id` or `target_project_id`.


**Note:** WebService supports only the AutoAI deployment type.



```
from ibm_watsonx_ai.deployment import WebService

service = WebService(wx_credentials,
     source_space_id='76g53e0-0b32-4a0e-9152-3d50324855ddb',
     target_space_id='1234abc1234abc1234abc1234abc1234abcd')
 )

service.create(
       experiment_run_id="...",
       model=model,
       deployment_name='My new deployment'
   )

```




Batch[¶](#batch "Link to this heading")
---------------------------------------


Batch manages the batch type of deployment. With Batch, you can upload and deploy a model and
run a batch deployment job. As with Web Service, you must pass the location where
the training was performed using the `source_space_id` or `source_project_id`.
You can deploy the model to any space or project by providing the `target_space_id` or `target_project_id`.


You can provide the input data as a `pandas.DataFrame`, a data-asset, or a Cloud Object Storage (COS) file.


**Note:** Batch supports only the AutoAI deployment type.


Example of a batch deployment creation:



```
from ibm_watsonx_ai.deployment import Batch

service_batch = Batch(wx_credentials, source_space_id='76g53e0-0b32-4a0e-9152-3d50324855ddb')
service_batch.create(
        experiment_run_id="6ce62a02-3e41-4d11-89d1-484c2deaed75",
        model="Pipeline_4",
        deployment_name='Batch deployment')

```


Example of a batch job creation with inline data as `pandas.DataFrame` type:



```
scoring_params = service_batch.run_job(
            payload=test_X_df,
            background_mode=False)

```


Example of batch job creation with a COS object:



```
from ibm_watsonx_ai.helpers.connections import S3Location, DataConnection

connection_details = client.connections.create({
    client.connections.ConfigurationMetaNames.NAME: "Connection to COS",
    client.connections.ConfigurationMetaNames.DATASOURCE_TYPE: client.connections.get_datasource_type_id_by_name('bluemixcloudobjectstorage'),
    client.connections.ConfigurationMetaNames.PROPERTIES: {
        'bucket': 'bucket_name',
        'access_key': 'COS access key id',
        'secret_key': 'COS secret access key'
        'iam_url': 'COS iam url',
        'url': 'COS endpoint url'
    }
})

connection_id = client.connections.get_uid(connection_details)

payload_reference = DataConnection(
        connection_asset_id=connection_id,
        location=S3Location(bucket='bucket_name',   # note: COS bucket name where deployment payload dataset is located
                            path='my_path'  # note: path within bucket where your deployment payload dataset is located
                            )
    )

results_reference = DataConnection(
        connection_asset_id=connection_id,
        location=S3Location(bucket='bucket_name',   # note: COS bucket name where deployment output should be located
                            path='my_path_where_output_will_be_saved'  # note: path within bucket where your deployment output should be located
                            )
    )
payload_reference.write("local_path_to_the_batch_payload_csv_file", remote_name="batch_payload_location.csv"])

scoring_params = service_batch.run_job(
    payload=[payload_reference],
    output_data_reference=results_reference,
    background_mode=False)   # If background_mode is False, then synchronous mode is started. Otherwise job status need to be monitored.

```


Example of a batch job creation with a data-asset object:



```
from ibm_watsonx_ai.helpers.connections import DataConnection, CloudAssetLocation, DeploymentOutputAssetLocation

payload_reference = DataConnection(location=CloudAssetLocation(asset_id=asset_id))
results_reference = DataConnection(
        location=DeploymentOutputAssetLocation(name="batch_output_file_name.csv"))

    scoring_params = service_batch.run_job(
        payload=[payload_reference],
        output_data_reference=results_reference,
        background_mode=False)

```








[Next

AutoAI experiment](autoai_experiment.html)
[Previous

AutoAI](autoai.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Working with AutoAI class and optimizer](#)
	+ [Configure optimizer with one data source](#configure-optimizer-with-one-data-source)
	+ [Configure optimizer for time series forecasting](#configure-optimizer-for-time-series-forecasting)
	+ [Configure optimizer for time series forecasting with supporting features](#configure-optimizer-for-time-series-forecasting-with-supporting-features)
	+ [Get configuration parameters](#get-configuration-parameters)
	+ [Fit optimizer](#fit-optimizer)
	+ [Get the run status and run details](#get-the-run-status-and-run-details)
	+ [Get data connections](#get-data-connections)
	+ [Pipeline summary](#pipeline-summary)
	+ [Get pipeline details](#get-pipeline-details)
	+ [Get pipeline](#get-pipeline)
	+ [Working with deployments](#working-with-deployments)
	+ [Web Service](#web-service)
	+ [Batch](#batch)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/dataconnection.html








DataConnection - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](#)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





DataConnection[¶](#dataconnection "Link to this heading")
=========================================================


DataConnection is the base class to start working with your data storage needed for AutoAI or Prompt Tuning experiment.
You can use it to fetch training data and store all of the results.



* [Working with DataConnection](autoai_working_with_dataconnection.html)
	+ [IBM Cloud - DataConnection Initialization](autoai_working_with_dataconnection.html#ibm-cloud-dataconnection-initialization)
		- [Connection Asset](autoai_working_with_dataconnection.html#connection-asset)
		- [Data Asset](autoai_working_with_dataconnection.html#data-asset)
		- [Container](autoai_working_with_dataconnection.html#container)
	+ [IBM watsonx.ai software - DataConnection Initialization](autoai_working_with_dataconnection.html#ibm-watsonx-ai-software-dataconnection-initialization)
		- [Connection Asset - DatabaseLocation](autoai_working_with_dataconnection.html#connection-asset-databaselocation)
		- [Connection Asset - S3Location](autoai_working_with_dataconnection.html#connection-asset-s3location)
		- [Connection Asset - NFSLocation](autoai_working_with_dataconnection.html#connection-asset-nfslocation)
		- [Data Asset](autoai_working_with_dataconnection.html#id1)
		- [FSLocation](autoai_working_with_dataconnection.html#fslocation)
	+ [Batch DataConnection](autoai_working_with_dataconnection.html#batch-dataconnection)
	+ [Upload your training dataset](autoai_working_with_dataconnection.html#upload-your-training-dataset)
	+ [Download your training dataset](autoai_working_with_dataconnection.html#download-your-training-dataset)
* [DataConnection Modules](dataconnection_modules.html)
	+ [DataConnection](dataconnection_modules.html#dataconnection)
		- [`DataConnection`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DataConnection)
			* [`DataConnection.from_studio()`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DataConnection.from_studio)
			* [`DataConnection.read()`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DataConnection.read)
			* [`DataConnection.set_client()`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DataConnection.set_client)
			* [`DataConnection.write()`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DataConnection.write)
	+ [S3Location](dataconnection_modules.html#s3location)
		- [`S3Location`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.S3Location)
			* [`S3Location.get_location()`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.S3Location.get_location)
	+ [CloudAssetLocation](dataconnection_modules.html#cloudassetlocation)
		- [`CloudAssetLocation`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.CloudAssetLocation)
	+ [DeploymentOutputAssetLocation](dataconnection_modules.html#deploymentoutputassetlocation)
		- [`DeploymentOutputAssetLocation`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DeploymentOutputAssetLocation)








[Next

Working with DataConnection](autoai_working_with_dataconnection.html)
[Previous

Federated Learning](federated_learning.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)
















# Text from https://ibm.github.io/watsonx-ai-python-sdk/install.html








Installation - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](#)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Installation[¶](#installation "Link to this heading")
=====================================================


The ibm-watsonx-ai Python library is publicly available
on PyPI: <https://pypi.org/project/ibm-watsonx-ai/>.


The package can be installed with pip:



```
$ pip install ibm-watsonx-ai

```



Note


The ibm-watsonx-ai Python library is available by default in all watsonx.ai notebook runtimes.




Product Offerings[¶](#product-offerings "Link to this heading")
---------------------------------------------------------------


The python package supports the following product offerings:


* [IBM watsonx.ai for IBM Cloud](https://dataplatform.cloud.ibm.com/docs/content/wsj/getting-started/welcome-main.html?audience=wdp&context=wx)
* IBM watsonx.ai software:
	+ [With IBM Cloud Pak for Data 4.8.x](https://www.ibm.com/docs/en/cloud-paks/cp-data/4.8.x)
	+ [With IBM Cloud Pak for Data 5.0.x](https://www.ibm.com/docs/en/cloud-paks/cp-data/5.0.x)



Note


If you are using Watson Machine Learning within IBM Cloud Pack for Data 4.0 - 4.7 versions, we recommend using ibm-watson-machine-learning package -
[see documentation.](file:///Users/mateuszszewczyk/Documents/GitHub/python-client/python/ibm_watson_machine_learning/docs/_build/html/install.html)



Differences in above product offerings are further described in section [Setup](setup.html).








[Next

Setup](setup.html)
[Previous

Home](index.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Installation](#)
	+ [Product Offerings](#product-offerings)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/prompt_tuner.html








Prompt Tuning - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](#)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Prompt Tuning[¶](#prompt-tuning "Link to this heading")
=======================================================


This version of `ibm-watsonx-ai` client introduces support for Tune Experiments.



* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)
	+ [Tune Experiment run](pt_tune_experiment_run.html)
		- [Configure PromptTuner](pt_tune_experiment_run.html#configure-prompttuner)
		- [Get configuration parameters](pt_tune_experiment_run.html#get-configuration-parameters)
		- [Run prompt tuning](pt_tune_experiment_run.html#run-prompt-tuning)
		- [Get run status, get run details](pt_tune_experiment_run.html#get-run-status-get-run-details)
		- [Get data connections](pt_tune_experiment_run.html#get-data-connections)
		- [Summary](pt_tune_experiment_run.html#summary)
		- [Plot learning curves](pt_tune_experiment_run.html#plot-learning-curves)
		- [Get model identifier](pt_tune_experiment_run.html#get-model-identifier)
	+ [Tuned Model Inference](pt_model_inference.html)
		- [Working with deployments](pt_model_inference.html#working-with-deployments)
		- [Creating `ModelInference` instance](pt_model_inference.html#creating-modelinference-instance)
		- [Importing data](pt_model_inference.html#importing-data)
		- [Analyzing satisfaction](pt_model_inference.html#analyzing-satisfaction)
		- [Generate methods](pt_model_inference.html#generate-methods)
* [Tune Experiment](tune_experiment.html)
	+ [TuneExperiment](tune_experiment.html#tuneexperiment)
		- [`TuneExperiment`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneExperiment)
			* [`TuneExperiment.prompt_tuner()`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneExperiment.prompt_tuner)
			* [`TuneExperiment.runs()`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneExperiment.runs)
	+ [Tune Runs](tune_experiment.html#tune-runs)
		- [`TuneRuns`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneRuns)
			* [`TuneRuns.get_run_details()`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneRuns.get_run_details)
			* [`TuneRuns.get_tuner()`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneRuns.get_tuner)
			* [`TuneRuns.list()`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneRuns.list)
	+ [Prompt Tuner](tune_experiment.html#prompt-tuner)
		- [`PromptTuner`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner)
			* [`PromptTuner.cancel_run()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.cancel_run)
			* [`PromptTuner.get_data_connections()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.get_data_connections)
			* [`PromptTuner.get_model_id()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.get_model_id)
			* [`PromptTuner.get_params()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.get_params)
			* [`PromptTuner.get_run_details()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.get_run_details)
			* [`PromptTuner.get_run_status()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.get_run_status)
			* [`PromptTuner.plot_learning_curve()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.plot_learning_curve)
			* [`PromptTuner.run()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.run)
			* [`PromptTuner.summary()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.summary)
	+ [Enums](tune_experiment.html#enums)
		- [`PromptTuningTypes`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningTypes)
			* [`PromptTuningTypes.PT`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningTypes.PT)
		- [`PromptTuningInitMethods`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningInitMethods)
			* [`PromptTuningInitMethods.RANDOM`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningInitMethods.RANDOM)
			* [`PromptTuningInitMethods.TEXT`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningInitMethods.TEXT)
		- [`TuneExperimentTasks`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks)
			* [`TuneExperimentTasks.CLASSIFICATION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.CLASSIFICATION)
			* [`TuneExperimentTasks.CODE_GENERATION_AND_CONVERSION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.CODE_GENERATION_AND_CONVERSION)
			* [`TuneExperimentTasks.EXTRACTION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.EXTRACTION)
			* [`TuneExperimentTasks.GENERATION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.GENERATION)
			* [`TuneExperimentTasks.QUESTION_ANSWERING`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.QUESTION_ANSWERING)
			* [`TuneExperimentTasks.RETRIEVAL_AUGMENTED_GENERATION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.RETRIEVAL_AUGMENTED_GENERATION)
			* [`TuneExperimentTasks.SUMMARIZATION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.SUMMARIZATION)
		- [`PromptTunableModels`](tune_experiment.html#PromptTunableModels)








[Next

Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)
[Previous

`ModelInference` for Deployments](fm_deployments.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)
















# Text from https://ibm.github.io/watsonx-ai-python-sdk/pt_tune_experiment_run.html








Tune Experiment run - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](#)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Tune Experiment run[¶](#tune-experiment-run "Link to this heading")
===================================================================


The [TuneExperiment class](tune_experiment.html#tune-experiment-class) is responsible for creating experiments and scheduling tunings.
All experiment results are stored automatically in the user-specified Cloud Object Storage (COS) for SaaS or
in cluster’s file system in case of Cloud Pak for Data. Then, the TuneExperiment feature can fetch the results and
provide them directly to the user for further usage.



Configure PromptTuner[¶](#configure-prompttuner "Link to this heading")
-----------------------------------------------------------------------


For an TuneExperiment object initialization authentication credentials (examples available in section: [Setup](setup.html)) and one of `project_id` or `space_id` are used.



Hint


You can copy the project\_id from the Project’s Manage tab (Project -> Manage -> General -> Details).




```
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.experiment import TuneExperiment

experiment = TuneExperiment(credentials,
    project_id="7ac03029-8bdd-4d5f-a561-2c4fd1e40705"
)

prompt_tuner = experiment.prompt_tuner(
    name="prompt tuning name",
    task_id=experiment.Tasks.CLASSIFICATION,
    base_model=ModelTypes.FLAN_T5_XL,
    accumulate_steps=32,
    batch_size=16,
    learning_rate=0.2,
    max_input_tokens=256,
    max_output_tokens=2,
    num_epochs=6,
    tuning_type=experiment.PromptTuningTypes.PT,
    verbalizer="Extract the satisfaction from the comment. Return simple '1' for satisfied customer or '0' for unsatisfied. Input: {{input}} Output: ",
    auto_update_model=True
)

```




Get configuration parameters[¶](#get-configuration-parameters "Link to this heading")
-------------------------------------------------------------------------------------


To see the current configuration parameters, call the `get_params()` method.



```
config_parameters = prompt_tuner.get_params()
print(config_parameters)
{
    'base_model': {'model_id': 'google/flan-t5-xl'},
    'accumulate_steps': 32,
    'batch_size': 16,
    'learning_rate': 0.2,
    'max_input_tokens': 256,
    'max_output_tokens': 2,
    'num_epochs': 6,
    'task_id': 'classification',
    'tuning_type': 'prompt_tuning',
    'verbalizer': "Extract the satisfaction from the comment. Return simple '1' for satisfied customer or '0' for unsatisfied. Input: {{input}} Output: ",
    'name': 'prompt tuning name',
    'description': 'Prompt tuning with SDK',
    'auto_update_model': True
}

```




Run prompt tuning[¶](#run-prompt-tuning "Link to this heading")
---------------------------------------------------------------


To schedule an tuning experiment, call the `run()` method (this will trigger a training process). The `run()` method can be synchronous (`background_mode=False`), or asynchronous (`background_mode=True`).
If you don’t want to wait for the training to end, invoke the async version. It immediately returns only run details.



```
from ibm_watsonx_ai.helpers import DataConnection, ContainerLocation, S3Location

tuning_details = prompt_tuner.run(
    training_data_references=[DataConnection(
        connection_asset_id=connection_id,
        location=S3Location(
            bucket='prompt_tuning_data',
            path='pt_train_data.json'
        )
    )],
    background_mode=False)

# OR

tuning_details = prompt_tuner.run(
    training_data_references=[DataConnection(
        data_asset_id='5d99c11a-2060-4ef6-83d5-dc593c6455e2')
    ],
    background_mode=True)

# OR

tuning_details = prompt_tuner.run(
    training_data_references=[DataConnection(
        location=ContainerLocation("path_to_file.json"))
    ],
    background_mode=True)

```




Get run status, get run details[¶](#get-run-status-get-run-details "Link to this heading")
------------------------------------------------------------------------------------------


If you use the `run()` method asynchronously, you can monitor the run details and status, using the following two methods:



```
status = prompt_tuner.get_run_status()
print(status)
'running'

# OR

'completed'

run_details = prompt_tuner.get_run_details()
print(run_details)
{
    'metadata': {'created_at': '2023-10-12T12:01:40.662Z',
    'description': 'Prompt tuning with SDK',
    'id': 'b3bc33b3-cb3f-49e7-9fb3-88c6c4d4f8d7',
    'modified_at': '2023-10-12T12:09:42.810Z',
    'name': 'prompt tuning name',
    'project_id': 'efa68764-5ec2-410a-bad9-982c502fbf4e',
    'tags': ['prompt_tuning',
    'wx_prompt_tune.3c06a0db-3cb9-478c-9421-eaf05276a1b7']},
    'entity': {'auto_update_model': True,
    'description': 'Prompt tuning with SDK',
    'model_id': 'd854752e-76a7-4c6d-b7db-5f84dd11e827',
    'name': 'prompt tuning name',
    'project_id': 'efa68764-5ec2-410a-bad9-982c502fbf4e',
    'prompt_tuning': {'accumulate_steps': 32,
    'base_model': {'model_id': 'google/flan-t5-xl'},
    'batch_size': 16,
    'init_method': 'random',
    'learning_rate': 0.2,
    'max_input_tokens': 256,
    'max_output_tokens': 2,
    'num_epochs': 6,
    'num_virtual_tokens': 100,
    'task_id': 'classification',
    'tuning_type': 'prompt_tuning',
    'verbalizer': "Extract the satisfaction from the comment. Return simple '1' for satisfied customer or '0' for unsatisfied. Input: {{input}} Output: "},
    'results_reference': {'connection': {},
    'location': {'path': 'default_tuning_output',
        'training': 'default_tuning_output/b3bc33b3-cb3f-49e7-9fb3-88c6c4d4f8d7',
        'training_status': 'default_tuning_output/b3bc33b3-cb3f-49e7-9fb3-88c6c4d4f8d7/training-status.json',
        'model_request_path': 'default_tuning_output/b3bc33b3-cb3f-49e7-9fb3-88c6c4d4f8d7/assets/b3bc33b3-cb3f-49e7-9fb3-88c6c4d4f8d7/resources/wml_model/request.json',
        'assets_path': 'default_tuning_output/b3bc33b3-cb3f-49e7-9fb3-88c6c4d4f8d7/assets'},
    'type': 'container'},
    'status': {'completed_at': '2023-10-12T12:09:42.769Z', 'state': 'completed'},
    'tags': ['prompt_tuning'],
    'training_data_references': [{'connection': {},
        'location': {'href': '/v2/assets/90258b10-5590-4d4c-be75-5eeeccf09076',
        'id': '90258b10-5590-4d4c-be75-5eeeccf09076'},
        'type': 'data_asset'}]}
}

```




Get data connections[¶](#get-data-connections "Link to this heading")
---------------------------------------------------------------------


The `data_connections` list contains all the training connections that you referenced while calling the `run()` method.



```
data_connections = prompt_tuner.get_data_connections()

# Get data in binary format
binary_data = data_connections[0].read(binary=True)

```




Summary[¶](#summary "Link to this heading")
-------------------------------------------


It is possible to see details of models in a form of summary table. The output type is a `pandas.DataFrame` with model names, enhancements, base model, auto update option, the number of epochs used and last loss function value.



```
results = prompt_tuner.summary()
print(results)

#                           Enhancements            Base model  ...         loss
#        Model Name
#  Prompt_tuned_M_1      [prompt_tuning]     google/flan-t5-xl  ...     0.449197

```




Plot learning curves[¶](#plot-learning-curves "Link to this heading")
---------------------------------------------------------------------



Note


Available only for Jupyter notebooks.



To see graphically how tuning was performed, you can view learning curve graphs.



```
prompt_tuner.plot_learning_curve()

```


[![_images/learning_curves.png](_images/learning_curves.png)](_images/learning_curves.png)


Get model identifier[¶](#get-model-identifier "Link to this heading")
---------------------------------------------------------------------



Note


It will be only available if the tuning was scheduled first and parameter `auto_update_model` was set as `True` (default value).



To get `model_id` call get\_model\_id method.



```
model_id = prompt_tuner.get_model_id()
print(model_id)
'd854752e-76a7-4c6d-b7db-5f84dd11e827'

```


The `model_id` obtained in this way can be used to create deployments and next create ModelInference.
For more information, see the next section: [Tuned Model Inference](pt_model_inference.html#pt-model-inference-module).








[Next

Tuned Model Inference](pt_model_inference.html)
[Previous

Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Tune Experiment run](#)
	+ [Configure PromptTuner](#configure-prompttuner)
	+ [Get configuration parameters](#get-configuration-parameters)
	+ [Run prompt tuning](#run-prompt-tuning)
	+ [Get run status, get run details](#get-run-status-get-run-details)
	+ [Get data connections](#get-data-connections)
	+ [Summary](#summary)
	+ [Plot learning curves](#plot-learning-curves)
	+ [Get model identifier](#get-model-identifier)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/fm_extensions.html








Extensions - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](#)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Extensions[¶](#extensions "Link to this heading")
=================================================



LangChain[¶](#langchain "Link to this heading")
-----------------------------------------------


IBM integration with langchain is available under [link](https://python.langchain.com/docs/integrations/providers/ibm).




*class* langchain\_ibm.WatsonxLLM(*\**, *name=None*, *cache=None*, *verbose=None*, *callbacks=None*, *tags=None*, *metadata=None*, *custom\_get\_token\_ids=None*, *callback\_manager=None*, *model\_id=''*, *deployment\_id=''*, *project\_id=''*, *space\_id=''*, *url=None*, *apikey=None*, *token=None*, *password=None*, *username=None*, *instance\_id=None*, *version=None*, *params=None*, *verify=None*, *streaming=False*, *watsonx\_model=None*)[[source]](_modules/langchain_ibm/llms.html#WatsonxLLM)[¶](#langchain_ibm.WatsonxLLM "Link to this definition")
IBM watsonx.ai large language models.


To use, you should have `langchain_ibm` python package installed,
and the environment variable `WATSONX_APIKEY` set with your API key, or pass
it as a named parameter to the constructor.



Example:
```
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
parameters = {
    GenTextParamsMetaNames.DECODING_METHOD: "sample",
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
    GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
    GenTextParamsMetaNames.TEMPERATURE: 0.5,
    GenTextParamsMetaNames.TOP_K: 50,
    GenTextParamsMetaNames.TOP_P: 1,
}

from langchain_ibm import WatsonxLLM
watsonx_llm = WatsonxLLM(
    model_id="google/flan-ul2",
    url="https://us-south.ml.cloud.ibm.com",
    apikey="*****",
    project_id="*****",
    params=parameters,
)

```






apikey[¶](#langchain_ibm.WatsonxLLM.apikey "Link to this definition")
Apikey to Watson Machine Learning or CPD instance





deployment\_id[¶](#langchain_ibm.WatsonxLLM.deployment_id "Link to this definition")
Type of deployed model to use.





get\_num\_tokens(*text*)[[source]](_modules/langchain_ibm/llms.html#WatsonxLLM.get_num_tokens)[¶](#langchain_ibm.WatsonxLLM.get_num_tokens "Link to this definition")
Get the number of tokens present in the text.


Useful for checking if an input will fit in a model’s context window.



Args:text: The string input to tokenize.



Returns:The integer number of tokens in the text.







get\_token\_ids(*text*)[[source]](_modules/langchain_ibm/llms.html#WatsonxLLM.get_token_ids)[¶](#langchain_ibm.WatsonxLLM.get_token_ids "Link to this definition")
Return the ordered ids of the tokens in a text.



Args:text: The string input to tokenize.



Returns:
A list of ids corresponding to the tokens in the text, in order they occurin the text.









instance\_id[¶](#langchain_ibm.WatsonxLLM.instance_id "Link to this definition")
Instance\_id of CPD instance





model\_id[¶](#langchain_ibm.WatsonxLLM.model_id "Link to this definition")
Type of model to use.





params[¶](#langchain_ibm.WatsonxLLM.params "Link to this definition")
Model parameters to use during generate requests.





password[¶](#langchain_ibm.WatsonxLLM.password "Link to this definition")
Password to CPD instance





project\_id[¶](#langchain_ibm.WatsonxLLM.project_id "Link to this definition")
ID of the Watson Studio project.





space\_id[¶](#langchain_ibm.WatsonxLLM.space_id "Link to this definition")
ID of the Watson Studio space.





streaming[¶](#langchain_ibm.WatsonxLLM.streaming "Link to this definition")
Whether to stream the results or not.





token[¶](#langchain_ibm.WatsonxLLM.token "Link to this definition")
Token to CPD instance





url[¶](#langchain_ibm.WatsonxLLM.url "Link to this definition")
Url to Watson Machine Learning or CPD instance





username[¶](#langchain_ibm.WatsonxLLM.username "Link to this definition")
Username to CPD instance





verify[¶](#langchain_ibm.WatsonxLLM.verify "Link to this definition")
User can pass as verify one of following:
the path to a CA\_BUNDLE file
the path of directory with certificates of trusted CAs
True - default path to truststore will be taken
False - no verification will be made





version[¶](#langchain_ibm.WatsonxLLM.version "Link to this definition")
Version of CPD instance




Example of **SimpleSequentialChain** usage



```
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

params = {
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1
}
credentials = Credentials(
                  url = "https://us-south.ml.cloud.ibm.com",
                  api_key = "***********"
                 )
project = "*****"

pt1 = PromptTemplate(
    input_variables=["topic"],
    template="Generate a random question about {topic}: Question: ")
pt2 = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question: {question}")

flan_ul2_model = ModelInference(
    model_id='google/flan-ul2',
    credentials=credentials,
    params=params,
    project_id=project_id)
flan_ul2_llm = WatsonxLLM(watsonx_model=flan_ul2_model)

flan_t5_model = ModelInference(
    model_id="google/flan-t5-xxl",
    credentials=credentials,
    project_id=project_id)
flan_t5_llm = WatsonxLLM(watsonx_model=flan_t5_model)

prompt_to_flan_ul2 = LLMChain(llm=flan_ul2_llm, prompt=pt1)
flan_ul2_to_flan_t5 = LLMChain(llm=flan_t5_llm, prompt=pt2)

qa = SimpleSequentialChain(chains=[prompt_to_flan_ul2, flan_ul2_to_flan_t5], verbose=True)
qa.run("cat")

```








[Next

Helpers](fm_helpers.html)
[Previous

Prompt Template Manager](prompt_template_manager.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Extensions](#)
	+ [LangChain](#langchain)
		- [`WatsonxLLM`](#langchain_ibm.WatsonxLLM)
			* [`WatsonxLLM.apikey`](#langchain_ibm.WatsonxLLM.apikey)
			* [`WatsonxLLM.deployment_id`](#langchain_ibm.WatsonxLLM.deployment_id)
			* [`WatsonxLLM.get_num_tokens()`](#langchain_ibm.WatsonxLLM.get_num_tokens)
			* [`WatsonxLLM.get_token_ids()`](#langchain_ibm.WatsonxLLM.get_token_ids)
			* [`WatsonxLLM.instance_id`](#langchain_ibm.WatsonxLLM.instance_id)
			* [`WatsonxLLM.model_id`](#langchain_ibm.WatsonxLLM.model_id)
			* [`WatsonxLLM.params`](#langchain_ibm.WatsonxLLM.params)
			* [`WatsonxLLM.password`](#langchain_ibm.WatsonxLLM.password)
			* [`WatsonxLLM.project_id`](#langchain_ibm.WatsonxLLM.project_id)
			* [`WatsonxLLM.space_id`](#langchain_ibm.WatsonxLLM.space_id)
			* [`WatsonxLLM.streaming`](#langchain_ibm.WatsonxLLM.streaming)
			* [`WatsonxLLM.token`](#langchain_ibm.WatsonxLLM.token)
			* [`WatsonxLLM.url`](#langchain_ibm.WatsonxLLM.url)
			* [`WatsonxLLM.username`](#langchain_ibm.WatsonxLLM.username)
			* [`WatsonxLLM.verify`](#langchain_ibm.WatsonxLLM.verify)
			* [`WatsonxLLM.version`](#langchain_ibm.WatsonxLLM.version)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/fm_working_with_custom_models.html








Custom models - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](#)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Custom models[¶](#custom-models "Link to this heading")
=======================================================



Note


Available in version IBM watsonx.ai software with IBM Cloud Pak for Data 4.8.4 and higher.



This section shows how to list custom models specs, store & deploy model and use ModelInference module with created deployment.



Initialize APIClient object[¶](#initialize-apiclient-object "Link to this heading")
-----------------------------------------------------------------------------------


Initialize `APIClient` object if needed. More details about supported `APIClient` initialization can be found in [Setup](setup.html) section,



```
from ibm_watsonx_ai import APIClient

client = APIClient(credentials)
client.set.default_project(project_id=project_id)
# or client.set.default_space(space_id=space_id)

```




Listing models specification[¶](#listing-models-specification "Link to this heading")
-------------------------------------------------------------------------------------



Warning


The model needs to be explicitly stored & deployed in the repository to be used/listed.



To list available custom models on PVC use example below. To get specification of specific model provide `model_id`.



```
from ibm_watsonx_ai.foundation_models import get_custom_model_specs

get_custom_models_spec(api_client=client)
# OR
get_custom_models_spec(credentials=credentials)
# OR
get_custom_models_spec(api_client=client, model_id='mistralai/Mistral-7B-Instruct-v0.2')

```




Storing model in service repository[¶](#storing-model-in-service-repository "Link to this heading")
---------------------------------------------------------------------------------------------------


To store model as an asset in the repo, first create proper `metadata`.



```
sw_spec_id = client.software_specifications.get_id_by_name('watsonx-cfm-caikit-1.0')

metadata = {
    client.repository.ModelMetaNames.NAME: 'custom FM asset',
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id,
    client.repository.ModelMetaNames.TYPE: client.repository.ModelAssetTypes.CUSTOM_FOUNDATION_MODEL_1_0
}

```


After that, it is possible to store model using `client.repository.store_model()`.



```
stored_model_details = client.repository.store_model(model='mistralai/Mistral-7B-Instruct-v0.2', meta_props=metadata)

```


To get `id` of stored asset use the details obtained.



```
model_asset_id = client.repository.get_model_id(stored_model_details)

```


All stored models custom foundation models can be listed by client.repository.list() method with filtering by framework type.



```
client.repository.list(framework_filter='custom_foundation_model_1.0')

```




Defining hardware specification[¶](#defining-hardware-specification "Link to this heading")
-------------------------------------------------------------------------------------------


For deployment of stored custom foundation model a hardware specifications need to be defined.
You can use custom hardware specification or pre-defined T-shirt sizes.
`APIClient` has dedicated module to work with [Hardware Specifications](core_api.html#core-api-hardware-specification). Few key methods are:


* List all defined hardware specifications:



```
client.hardware_specifications.list()

```


* Retrieve details of defined hardware specifications:



```
client.hardware_specifications.get_details(client.hardware_specifications.get_id_by_name('M'))

```


* Define custom hardware specification:



```
meta_props = {
    client.hardware_specifications.ConfigurationMetaNames.NAME: "Custom GPU hw spec",
    client.hardware_specifications.ConfigurationMetaNames.NODES:{"cpu":{"units":"2"},"mem":{"size":"128Gi"},"gpu":{"num_gpu":1}}
    }

hw_spec_details = client.hardware_specifications.store(meta_props)

```




Deployment of custom foundation model[¶](#deployment-of-custom-foundation-model "Link to this heading")
-------------------------------------------------------------------------------------------------------


To crete new deployment of custom foundation models dictionary with deployment `metadata` need to be defined.
There can be specified the `NAME` of new deployment, `DESCRIPTION` and hardware specification.
For now only online deployments are supported so `ONLINE` field is required.
At this stage user can overwrite model parameters optionally.
It can be done by passing dictionary with new parameters values in `FOUNDATION_MODEL` field.


Besides the `metadata` with deployment configuration the `id` of stored model asset are required for deployment creation.



```
metadata = {
    client.deployments.ConfigurationMetaNames.NAME: "Custom FM Deployment",
    client.deployments.ConfigurationMetaNames.DESCRIPTION: "Deployment of custom foundation model with SDK",
    client.deployments.ConfigurationMetaNames.ONLINE: {},
    client.deployments.ConfigurationMetaNames.HARDWARE_SPEC : { "name":  "Custom GPU hw spec"}, # name or id supported here
    client.deployments.ConfigurationMetaNames.FOUNDATION_MODEL: {"max_new_tokens": 128}.  # optional
}
deployment_details = client.deployments.create(model_asset_id, metadata)

```


Once deployment creation process is done the `client.deployments.create` returns dictionary with deployment details,
which can be used to retrieve the `id` of the deployment.



```
deployment_id = client.deployments.get_id(deployment_details)

```


All existing in working space or project scope can be listed with `list` method:



```
client.deployments.list()

```




Working with deployments[¶](#working-with-deployments "Link to this heading")
-----------------------------------------------------------------------------


Working with deployments of foundation models is described in section [Models/ ModelInference for Deployments](fm_deployments.html).








[Next

Samples](samples.html)
[Previous

Helpers](fm_helpers.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Custom models](#)
	+ [Initialize APIClient object](#initialize-apiclient-object)
	+ [Listing models specification](#listing-models-specification)
	+ [Storing model in service repository](#storing-model-in-service-repository)
	+ [Defining hardware specification](#defining-hardware-specification)
	+ [Deployment of custom foundation model](#deployment-of-custom-foundation-model)
	+ [Working with deployments](#working-with-deployments)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/api.html








API - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](#)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





API[¶](#api "Link to this heading")
===================================


`ibm-watsonx-ai` client contains different APIs available for AutoAI and assets management.



Prerequisites[¶](#prerequisites "Link to this heading")
-------------------------------------------------------


Initialize the client using the watsonx.ai instance details provided in [Setup](setup.html).




Modules[¶](#modules "Link to this heading")
-------------------------------------------



* [Base](base.html)
	+ [APIClient](base.html#apiclient)
		- [`APIClient`](base.html#client.APIClient)
			* [`APIClient.set_headers()`](base.html#client.APIClient.set_headers)
			* [`APIClient.set_token()`](base.html#client.APIClient.set_token)
	+ [Credentials](base.html#credentials)
		- [`Credentials`](base.html#credentials.Credentials)
			* [`Credentials.from_dict()`](base.html#credentials.Credentials.from_dict)
			* [`Credentials.to_dict()`](base.html#credentials.Credentials.to_dict)
* [Core](core_api.html)
	+ [Connections](core_api.html#connections)
		- [`Connections`](core_api.html#client.Connections)
			* [`Connections.ConfigurationMetaNames`](core_api.html#client.Connections.ConfigurationMetaNames)
			* [`Connections.create()`](core_api.html#client.Connections.create)
			* [`Connections.delete()`](core_api.html#client.Connections.delete)
			* [`Connections.get_datasource_type_details_by_id()`](core_api.html#client.Connections.get_datasource_type_details_by_id)
			* [`Connections.get_datasource_type_id_by_name()`](core_api.html#client.Connections.get_datasource_type_id_by_name)
			* [`Connections.get_datasource_type_uid_by_name()`](core_api.html#client.Connections.get_datasource_type_uid_by_name)
			* [`Connections.get_details()`](core_api.html#client.Connections.get_details)
			* [`Connections.get_id()`](core_api.html#client.Connections.get_id)
			* [`Connections.get_uid()`](core_api.html#client.Connections.get_uid)
			* [`Connections.get_uploaded_db_drivers()`](core_api.html#client.Connections.get_uploaded_db_drivers)
			* [`Connections.list()`](core_api.html#client.Connections.list)
			* [`Connections.list_datasource_types()`](core_api.html#client.Connections.list_datasource_types)
			* [`Connections.list_uploaded_db_drivers()`](core_api.html#client.Connections.list_uploaded_db_drivers)
			* [`Connections.sign_db_driver_url()`](core_api.html#client.Connections.sign_db_driver_url)
			* [`Connections.upload_db_driver()`](core_api.html#client.Connections.upload_db_driver)
		- [`ConnectionMetaNames`](core_api.html#metanames.ConnectionMetaNames)
	+ [Data assets](core_api.html#data-assets)
		- [`Assets`](core_api.html#client.Assets)
			* [`Assets.ConfigurationMetaNames`](core_api.html#client.Assets.ConfigurationMetaNames)
			* [`Assets.create()`](core_api.html#client.Assets.create)
			* [`Assets.delete()`](core_api.html#client.Assets.delete)
			* [`Assets.download()`](core_api.html#client.Assets.download)
			* [`Assets.get_content()`](core_api.html#client.Assets.get_content)
			* [`Assets.get_details()`](core_api.html#client.Assets.get_details)
			* [`Assets.get_href()`](core_api.html#client.Assets.get_href)
			* [`Assets.get_id()`](core_api.html#client.Assets.get_id)
			* [`Assets.list()`](core_api.html#client.Assets.list)
			* [`Assets.store()`](core_api.html#client.Assets.store)
		- [`AssetsMetaNames`](core_api.html#metanames.AssetsMetaNames)
	+ [Deployments](core_api.html#deployments)
		- [`Deployments`](core_api.html#client.Deployments)
			* [`Deployments.create()`](core_api.html#client.Deployments.create)
			* [`Deployments.create_job()`](core_api.html#client.Deployments.create_job)
			* [`Deployments.delete()`](core_api.html#client.Deployments.delete)
			* [`Deployments.delete_job()`](core_api.html#client.Deployments.delete_job)
			* [`Deployments.generate()`](core_api.html#client.Deployments.generate)
			* [`Deployments.generate_text()`](core_api.html#client.Deployments.generate_text)
			* [`Deployments.generate_text_stream()`](core_api.html#client.Deployments.generate_text_stream)
			* [`Deployments.get_details()`](core_api.html#client.Deployments.get_details)
			* [`Deployments.get_download_url()`](core_api.html#client.Deployments.get_download_url)
			* [`Deployments.get_href()`](core_api.html#client.Deployments.get_href)
			* [`Deployments.get_id()`](core_api.html#client.Deployments.get_id)
			* [`Deployments.get_job_details()`](core_api.html#client.Deployments.get_job_details)
			* [`Deployments.get_job_href()`](core_api.html#client.Deployments.get_job_href)
			* [`Deployments.get_job_id()`](core_api.html#client.Deployments.get_job_id)
			* [`Deployments.get_job_status()`](core_api.html#client.Deployments.get_job_status)
			* [`Deployments.get_job_uid()`](core_api.html#client.Deployments.get_job_uid)
			* [`Deployments.get_scoring_href()`](core_api.html#client.Deployments.get_scoring_href)
			* [`Deployments.get_serving_href()`](core_api.html#client.Deployments.get_serving_href)
			* [`Deployments.get_uid()`](core_api.html#client.Deployments.get_uid)
			* [`Deployments.is_serving_name_available()`](core_api.html#client.Deployments.is_serving_name_available)
			* [`Deployments.list()`](core_api.html#client.Deployments.list)
			* [`Deployments.list_jobs()`](core_api.html#client.Deployments.list_jobs)
			* [`Deployments.score()`](core_api.html#client.Deployments.score)
			* [`Deployments.update()`](core_api.html#client.Deployments.update)
		- [`DeploymentMetaNames`](core_api.html#metanames.DeploymentMetaNames)
		- [`RShinyAuthenticationValues`](core_api.html#ibm_watsonx_ai.utils.enums.RShinyAuthenticationValues)
			* [`RShinyAuthenticationValues.ANYONE_WITH_URL`](core_api.html#ibm_watsonx_ai.utils.enums.RShinyAuthenticationValues.ANYONE_WITH_URL)
			* [`RShinyAuthenticationValues.ANY_VALID_USER`](core_api.html#ibm_watsonx_ai.utils.enums.RShinyAuthenticationValues.ANY_VALID_USER)
			* [`RShinyAuthenticationValues.MEMBERS_OF_DEPLOYMENT_SPACE`](core_api.html#ibm_watsonx_ai.utils.enums.RShinyAuthenticationValues.MEMBERS_OF_DEPLOYMENT_SPACE)
		- [`ScoringMetaNames`](core_api.html#metanames.ScoringMetaNames)
		- [`DecisionOptimizationMetaNames`](core_api.html#metanames.DecisionOptimizationMetaNames)
	+ [Export/Import](core_api.html#export-import)
		- [`Export`](core_api.html#client.Export)
			* [`Export.cancel()`](core_api.html#client.Export.cancel)
			* [`Export.delete()`](core_api.html#client.Export.delete)
			* [`Export.get_details()`](core_api.html#client.Export.get_details)
			* [`Export.get_exported_content()`](core_api.html#client.Export.get_exported_content)
			* [`Export.get_id()`](core_api.html#client.Export.get_id)
			* [`Export.list()`](core_api.html#client.Export.list)
			* [`Export.start()`](core_api.html#client.Export.start)
		- [`Import`](core_api.html#client.Import)
			* [`Import.cancel()`](core_api.html#client.Import.cancel)
			* [`Import.delete()`](core_api.html#client.Import.delete)
			* [`Import.get_details()`](core_api.html#client.Import.get_details)
			* [`Import.get_id()`](core_api.html#client.Import.get_id)
			* [`Import.list()`](core_api.html#client.Import.list)
			* [`Import.start()`](core_api.html#client.Import.start)
	+ [Factsheets (IBM Cloud only)](core_api.html#factsheets-ibm-cloud-only)
		- [`Factsheets`](core_api.html#client.Factsheets)
			* [`Factsheets.list_model_entries()`](core_api.html#client.Factsheets.list_model_entries)
			* [`Factsheets.register_model_entry()`](core_api.html#client.Factsheets.register_model_entry)
			* [`Factsheets.unregister_model_entry()`](core_api.html#client.Factsheets.unregister_model_entry)
		- [`FactsheetsMetaNames`](core_api.html#metanames.FactsheetsMetaNames)
	+ [Hardware specifications](core_api.html#hardware-specifications)
		- [`HwSpec`](core_api.html#client.HwSpec)
			* [`HwSpec.ConfigurationMetaNames`](core_api.html#client.HwSpec.ConfigurationMetaNames)
			* [`HwSpec.delete()`](core_api.html#client.HwSpec.delete)
			* [`HwSpec.get_details()`](core_api.html#client.HwSpec.get_details)
			* [`HwSpec.get_href()`](core_api.html#client.HwSpec.get_href)
			* [`HwSpec.get_id()`](core_api.html#client.HwSpec.get_id)
			* [`HwSpec.get_id_by_name()`](core_api.html#client.HwSpec.get_id_by_name)
			* [`HwSpec.get_uid()`](core_api.html#client.HwSpec.get_uid)
			* [`HwSpec.get_uid_by_name()`](core_api.html#client.HwSpec.get_uid_by_name)
			* [`HwSpec.list()`](core_api.html#client.HwSpec.list)
			* [`HwSpec.store()`](core_api.html#client.HwSpec.store)
		- [`HwSpecMetaNames`](core_api.html#metanames.HwSpecMetaNames)
	+ [Helpers](core_api.html#helpers)
		- [`get_credentials_from_config`](core_api.html#ibm_watsonx_ai.helpers.helpers.get_credentials_from_config)
	+ [Model definitions](core_api.html#model-definitions)
		- [`ModelDefinition`](core_api.html#client.ModelDefinition)
			* [`ModelDefinition.ConfigurationMetaNames`](core_api.html#client.ModelDefinition.ConfigurationMetaNames)
			* [`ModelDefinition.create_revision()`](core_api.html#client.ModelDefinition.create_revision)
			* [`ModelDefinition.delete()`](core_api.html#client.ModelDefinition.delete)
			* [`ModelDefinition.download()`](core_api.html#client.ModelDefinition.download)
			* [`ModelDefinition.get_details()`](core_api.html#client.ModelDefinition.get_details)
			* [`ModelDefinition.get_href()`](core_api.html#client.ModelDefinition.get_href)
			* [`ModelDefinition.get_id()`](core_api.html#client.ModelDefinition.get_id)
			* [`ModelDefinition.get_revision_details()`](core_api.html#client.ModelDefinition.get_revision_details)
			* [`ModelDefinition.get_uid()`](core_api.html#client.ModelDefinition.get_uid)
			* [`ModelDefinition.list()`](core_api.html#client.ModelDefinition.list)
			* [`ModelDefinition.list_revisions()`](core_api.html#client.ModelDefinition.list_revisions)
			* [`ModelDefinition.store()`](core_api.html#client.ModelDefinition.store)
			* [`ModelDefinition.update()`](core_api.html#client.ModelDefinition.update)
		- [`ModelDefinitionMetaNames`](core_api.html#metanames.ModelDefinitionMetaNames)
	+ [Package extensions](core_api.html#package-extensions)
		- [`PkgExtn`](core_api.html#client.PkgExtn)
			* [`PkgExtn.ConfigurationMetaNames`](core_api.html#client.PkgExtn.ConfigurationMetaNames)
			* [`PkgExtn.delete()`](core_api.html#client.PkgExtn.delete)
			* [`PkgExtn.download()`](core_api.html#client.PkgExtn.download)
			* [`PkgExtn.get_details()`](core_api.html#client.PkgExtn.get_details)
			* [`PkgExtn.get_href()`](core_api.html#client.PkgExtn.get_href)
			* [`PkgExtn.get_id()`](core_api.html#client.PkgExtn.get_id)
			* [`PkgExtn.get_id_by_name()`](core_api.html#client.PkgExtn.get_id_by_name)
			* [`PkgExtn.list()`](core_api.html#client.PkgExtn.list)
			* [`PkgExtn.store()`](core_api.html#client.PkgExtn.store)
		- [`PkgExtnMetaNames`](core_api.html#metanames.PkgExtnMetaNames)
	+ [Parameter Sets](core_api.html#parameter-sets)
		- [`ParameterSets`](core_api.html#client.ParameterSets)
			* [`ParameterSets.ConfigurationMetaNames`](core_api.html#client.ParameterSets.ConfigurationMetaNames)
			* [`ParameterSets.create()`](core_api.html#client.ParameterSets.create)
			* [`ParameterSets.delete()`](core_api.html#client.ParameterSets.delete)
			* [`ParameterSets.get_details()`](core_api.html#client.ParameterSets.get_details)
			* [`ParameterSets.get_id_by_name()`](core_api.html#client.ParameterSets.get_id_by_name)
			* [`ParameterSets.list()`](core_api.html#client.ParameterSets.list)
			* [`ParameterSets.update()`](core_api.html#client.ParameterSets.update)
		- [`ParameterSetsMetaNames`](core_api.html#metanames.ParameterSetsMetaNames)
	+ [Repository](core_api.html#repository)
		- [`Repository`](core_api.html#client.Repository)
			* [`Repository.ModelAssetTypes`](core_api.html#client.Repository.ModelAssetTypes)
			* [`Repository.create_experiment_revision()`](core_api.html#client.Repository.create_experiment_revision)
			* [`Repository.create_function_revision()`](core_api.html#client.Repository.create_function_revision)
			* [`Repository.create_model_revision()`](core_api.html#client.Repository.create_model_revision)
			* [`Repository.create_pipeline_revision()`](core_api.html#client.Repository.create_pipeline_revision)
			* [`Repository.create_revision()`](core_api.html#client.Repository.create_revision)
			* [`Repository.delete()`](core_api.html#client.Repository.delete)
			* [`Repository.download()`](core_api.html#client.Repository.download)
			* [`Repository.get_details()`](core_api.html#client.Repository.get_details)
			* [`Repository.get_experiment_details()`](core_api.html#client.Repository.get_experiment_details)
			* [`Repository.get_experiment_href()`](core_api.html#client.Repository.get_experiment_href)
			* [`Repository.get_experiment_id()`](core_api.html#client.Repository.get_experiment_id)
			* [`Repository.get_experiment_revision_details()`](core_api.html#client.Repository.get_experiment_revision_details)
			* [`Repository.get_function_details()`](core_api.html#client.Repository.get_function_details)
			* [`Repository.get_function_href()`](core_api.html#client.Repository.get_function_href)
			* [`Repository.get_function_id()`](core_api.html#client.Repository.get_function_id)
			* [`Repository.get_function_revision_details()`](core_api.html#client.Repository.get_function_revision_details)
			* [`Repository.get_model_details()`](core_api.html#client.Repository.get_model_details)
			* [`Repository.get_model_href()`](core_api.html#client.Repository.get_model_href)
			* [`Repository.get_model_id()`](core_api.html#client.Repository.get_model_id)
			* [`Repository.get_model_revision_details()`](core_api.html#client.Repository.get_model_revision_details)
			* [`Repository.get_pipeline_details()`](core_api.html#client.Repository.get_pipeline_details)
			* [`Repository.get_pipeline_href()`](core_api.html#client.Repository.get_pipeline_href)
			* [`Repository.get_pipeline_id()`](core_api.html#client.Repository.get_pipeline_id)
			* [`Repository.get_pipeline_revision_details()`](core_api.html#client.Repository.get_pipeline_revision_details)
			* [`Repository.list()`](core_api.html#client.Repository.list)
			* [`Repository.list_experiments()`](core_api.html#client.Repository.list_experiments)
			* [`Repository.list_experiments_revisions()`](core_api.html#client.Repository.list_experiments_revisions)
			* [`Repository.list_functions()`](core_api.html#client.Repository.list_functions)
			* [`Repository.list_functions_revisions()`](core_api.html#client.Repository.list_functions_revisions)
			* [`Repository.list_models()`](core_api.html#client.Repository.list_models)
			* [`Repository.list_models_revisions()`](core_api.html#client.Repository.list_models_revisions)
			* [`Repository.list_pipelines()`](core_api.html#client.Repository.list_pipelines)
			* [`Repository.list_pipelines_revisions()`](core_api.html#client.Repository.list_pipelines_revisions)
			* [`Repository.load()`](core_api.html#client.Repository.load)
			* [`Repository.promote_model()`](core_api.html#client.Repository.promote_model)
			* [`Repository.store_experiment()`](core_api.html#client.Repository.store_experiment)
			* [`Repository.store_function()`](core_api.html#client.Repository.store_function)
			* [`Repository.store_model()`](core_api.html#client.Repository.store_model)
			* [`Repository.store_pipeline()`](core_api.html#client.Repository.store_pipeline)
			* [`Repository.update_experiment()`](core_api.html#client.Repository.update_experiment)
			* [`Repository.update_function()`](core_api.html#client.Repository.update_function)
			* [`Repository.update_model()`](core_api.html#client.Repository.update_model)
			* [`Repository.update_pipeline()`](core_api.html#client.Repository.update_pipeline)
		- [`ModelMetaNames`](core_api.html#metanames.ModelMetaNames)
		- [`ExperimentMetaNames`](core_api.html#metanames.ExperimentMetaNames)
		- [`FunctionMetaNames`](core_api.html#metanames.FunctionMetaNames)
		- [`PipelineMetanames`](core_api.html#metanames.PipelineMetanames)
	+ [Script](core_api.html#script)
		- [`Script`](core_api.html#client.Script)
			* [`Script.ConfigurationMetaNames`](core_api.html#client.Script.ConfigurationMetaNames)
			* [`Script.create_revision()`](core_api.html#client.Script.create_revision)
			* [`Script.delete()`](core_api.html#client.Script.delete)
			* [`Script.download()`](core_api.html#client.Script.download)
			* [`Script.get_details()`](core_api.html#client.Script.get_details)
			* [`Script.get_href()`](core_api.html#client.Script.get_href)
			* [`Script.get_id()`](core_api.html#client.Script.get_id)
			* [`Script.get_revision_details()`](core_api.html#client.Script.get_revision_details)
			* [`Script.list()`](core_api.html#client.Script.list)
			* [`Script.list_revisions()`](core_api.html#client.Script.list_revisions)
			* [`Script.store()`](core_api.html#client.Script.store)
			* [`Script.update()`](core_api.html#client.Script.update)
		- [`ScriptMetaNames`](core_api.html#metanames.ScriptMetaNames)
	+ [Service instance](core_api.html#service-instance)
		- [`ServiceInstance`](core_api.html#client.ServiceInstance)
			* [`ServiceInstance.get_api_key()`](core_api.html#client.ServiceInstance.get_api_key)
			* [`ServiceInstance.get_details()`](core_api.html#client.ServiceInstance.get_details)
			* [`ServiceInstance.get_instance_id()`](core_api.html#client.ServiceInstance.get_instance_id)
			* [`ServiceInstance.get_password()`](core_api.html#client.ServiceInstance.get_password)
			* [`ServiceInstance.get_url()`](core_api.html#client.ServiceInstance.get_url)
			* [`ServiceInstance.get_username()`](core_api.html#client.ServiceInstance.get_username)
	+ [Set](core_api.html#set)
		- [`Set`](core_api.html#client.Set)
			* [`Set.default_project()`](core_api.html#client.Set.default_project)
			* [`Set.default_space()`](core_api.html#client.Set.default_space)
	+ [Shiny (IBM Cloud Pak for Data only)](core_api.html#shiny-ibm-cloud-pak-for-data-only)
		- [`Shiny`](core_api.html#client.Shiny)
			* [`Shiny.ConfigurationMetaNames`](core_api.html#client.Shiny.ConfigurationMetaNames)
			* [`Shiny.create_revision()`](core_api.html#client.Shiny.create_revision)
			* [`Shiny.delete()`](core_api.html#client.Shiny.delete)
			* [`Shiny.download()`](core_api.html#client.Shiny.download)
			* [`Shiny.get_details()`](core_api.html#client.Shiny.get_details)
			* [`Shiny.get_href()`](core_api.html#client.Shiny.get_href)
			* [`Shiny.get_id()`](core_api.html#client.Shiny.get_id)
			* [`Shiny.get_revision_details()`](core_api.html#client.Shiny.get_revision_details)
			* [`Shiny.get_uid()`](core_api.html#client.Shiny.get_uid)
			* [`Shiny.list()`](core_api.html#client.Shiny.list)
			* [`Shiny.list_revisions()`](core_api.html#client.Shiny.list_revisions)
			* [`Shiny.store()`](core_api.html#client.Shiny.store)
			* [`Shiny.update()`](core_api.html#client.Shiny.update)
	+ [Software specifications](core_api.html#software-specifications)
		- [`SwSpec`](core_api.html#client.SwSpec)
			* [`SwSpec.ConfigurationMetaNames`](core_api.html#client.SwSpec.ConfigurationMetaNames)
			* [`SwSpec.add_package_extension()`](core_api.html#client.SwSpec.add_package_extension)
			* [`SwSpec.delete()`](core_api.html#client.SwSpec.delete)
			* [`SwSpec.delete_package_extension()`](core_api.html#client.SwSpec.delete_package_extension)
			* [`SwSpec.get_details()`](core_api.html#client.SwSpec.get_details)
			* [`SwSpec.get_href()`](core_api.html#client.SwSpec.get_href)
			* [`SwSpec.get_id()`](core_api.html#client.SwSpec.get_id)
			* [`SwSpec.get_id_by_name()`](core_api.html#client.SwSpec.get_id_by_name)
			* [`SwSpec.get_uid()`](core_api.html#client.SwSpec.get_uid)
			* [`SwSpec.get_uid_by_name()`](core_api.html#client.SwSpec.get_uid_by_name)
			* [`SwSpec.list()`](core_api.html#client.SwSpec.list)
			* [`SwSpec.store()`](core_api.html#client.SwSpec.store)
		- [`SwSpecMetaNames`](core_api.html#metanames.SwSpecMetaNames)
	+ [Spaces](core_api.html#spaces)
		- [`Spaces`](core_api.html#client.Spaces)
			* [`Spaces.ConfigurationMetaNames`](core_api.html#client.Spaces.ConfigurationMetaNames)
			* [`Spaces.MemberMetaNames`](core_api.html#client.Spaces.MemberMetaNames)
			* [`Spaces.create_member()`](core_api.html#client.Spaces.create_member)
			* [`Spaces.delete()`](core_api.html#client.Spaces.delete)
			* [`Spaces.delete_member()`](core_api.html#client.Spaces.delete_member)
			* [`Spaces.get_details()`](core_api.html#client.Spaces.get_details)
			* [`Spaces.get_id()`](core_api.html#client.Spaces.get_id)
			* [`Spaces.get_member_details()`](core_api.html#client.Spaces.get_member_details)
			* [`Spaces.get_uid()`](core_api.html#client.Spaces.get_uid)
			* [`Spaces.list()`](core_api.html#client.Spaces.list)
			* [`Spaces.list_members()`](core_api.html#client.Spaces.list_members)
			* [`Spaces.promote()`](core_api.html#client.Spaces.promote)
			* [`Spaces.store()`](core_api.html#client.Spaces.store)
			* [`Spaces.update()`](core_api.html#client.Spaces.update)
			* [`Spaces.update_member()`](core_api.html#client.Spaces.update_member)
		- [`SpacesMetaNames`](core_api.html#metanames.SpacesMetaNames)
		- [`SpacesMemberMetaNames`](core_api.html#metanames.SpacesMemberMetaNames)
	+ [Training](core_api.html#training)
		- [`Training`](core_api.html#client.Training)
			* [`Training.cancel()`](core_api.html#client.Training.cancel)
			* [`Training.get_details()`](core_api.html#client.Training.get_details)
			* [`Training.get_href()`](core_api.html#client.Training.get_href)
			* [`Training.get_id()`](core_api.html#client.Training.get_id)
			* [`Training.get_metrics()`](core_api.html#client.Training.get_metrics)
			* [`Training.get_status()`](core_api.html#client.Training.get_status)
			* [`Training.list()`](core_api.html#client.Training.list)
			* [`Training.list_intermediate_models()`](core_api.html#client.Training.list_intermediate_models)
			* [`Training.monitor_logs()`](core_api.html#client.Training.monitor_logs)
			* [`Training.monitor_metrics()`](core_api.html#client.Training.monitor_metrics)
			* [`Training.run()`](core_api.html#client.Training.run)
		- [`TrainingConfigurationMetaNames`](core_api.html#metanames.TrainingConfigurationMetaNames)
	+ [Enums](core_api.html#module-ibm_watsonx_ai.utils.autoai.enums)
		- [`ClassificationAlgorithms`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms)
			* [`ClassificationAlgorithms.DT`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.DT)
			* [`ClassificationAlgorithms.EX_TREES`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.EX_TREES)
			* [`ClassificationAlgorithms.GB`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.GB)
			* [`ClassificationAlgorithms.LGBM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.LGBM)
			* [`ClassificationAlgorithms.LR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.LR)
			* [`ClassificationAlgorithms.RF`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.RF)
			* [`ClassificationAlgorithms.SnapBM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.SnapBM)
			* [`ClassificationAlgorithms.SnapDT`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.SnapDT)
			* [`ClassificationAlgorithms.SnapLR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.SnapLR)
			* [`ClassificationAlgorithms.SnapRF`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.SnapRF)
			* [`ClassificationAlgorithms.SnapSVM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.SnapSVM)
			* [`ClassificationAlgorithms.XGB`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithms.XGB)
		- [`ClassificationAlgorithmsCP4D`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D)
			* [`ClassificationAlgorithmsCP4D.DT`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.DT)
			* [`ClassificationAlgorithmsCP4D.EX_TREES`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.EX_TREES)
			* [`ClassificationAlgorithmsCP4D.GB`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.GB)
			* [`ClassificationAlgorithmsCP4D.LGBM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.LGBM)
			* [`ClassificationAlgorithmsCP4D.LR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.LR)
			* [`ClassificationAlgorithmsCP4D.RF`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.RF)
			* [`ClassificationAlgorithmsCP4D.SnapBM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.SnapBM)
			* [`ClassificationAlgorithmsCP4D.SnapDT`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.SnapDT)
			* [`ClassificationAlgorithmsCP4D.SnapLR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.SnapLR)
			* [`ClassificationAlgorithmsCP4D.SnapRF`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.SnapRF)
			* [`ClassificationAlgorithmsCP4D.SnapSVM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.SnapSVM)
			* [`ClassificationAlgorithmsCP4D.XGB`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ClassificationAlgorithmsCP4D.XGB)
		- [`DataConnectionTypes`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes)
			* [`DataConnectionTypes.CA`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes.CA)
			* [`DataConnectionTypes.CN`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes.CN)
			* [`DataConnectionTypes.DS`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes.DS)
			* [`DataConnectionTypes.FS`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes.FS)
			* [`DataConnectionTypes.S3`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.DataConnectionTypes.S3)
		- [`Directions`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Directions)
			* [`Directions.ASCENDING`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Directions.ASCENDING)
			* [`Directions.DESCENDING`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Directions.DESCENDING)
		- [`ForecastingAlgorithms`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms)
			* [`ForecastingAlgorithms.ARIMA`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.ARIMA)
			* [`ForecastingAlgorithms.BATS`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.BATS)
			* [`ForecastingAlgorithms.ENSEMBLER`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.ENSEMBLER)
			* [`ForecastingAlgorithms.HW`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.HW)
			* [`ForecastingAlgorithms.LR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.LR)
			* [`ForecastingAlgorithms.RF`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.RF)
			* [`ForecastingAlgorithms.SVM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithms.SVM)
		- [`ForecastingAlgorithmsCP4D`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D)
			* [`ForecastingAlgorithmsCP4D.ARIMA`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.ARIMA)
			* [`ForecastingAlgorithmsCP4D.BATS`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.BATS)
			* [`ForecastingAlgorithmsCP4D.ENSEMBLER`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.ENSEMBLER)
			* [`ForecastingAlgorithmsCP4D.HW`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.HW)
			* [`ForecastingAlgorithmsCP4D.LR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.LR)
			* [`ForecastingAlgorithmsCP4D.RF`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.RF)
			* [`ForecastingAlgorithmsCP4D.SVM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingAlgorithmsCP4D.SVM)
		- [`ForecastingPipelineTypes`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes)
			* [`ForecastingPipelineTypes.ARIMA`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMA)
			* [`ForecastingPipelineTypes.ARIMAX`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMAX)
			* [`ForecastingPipelineTypes.ARIMAX_DMLR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMAX_DMLR)
			* [`ForecastingPipelineTypes.ARIMAX_PALR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMAX_PALR)
			* [`ForecastingPipelineTypes.ARIMAX_RAR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMAX_RAR)
			* [`ForecastingPipelineTypes.ARIMAX_RSAR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ARIMAX_RSAR)
			* [`ForecastingPipelineTypes.Bats`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.Bats)
			* [`ForecastingPipelineTypes.DifferenceFlattenEnsembler`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.DifferenceFlattenEnsembler)
			* [`ForecastingPipelineTypes.ExogenousDifferenceFlattenEnsembler`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousDifferenceFlattenEnsembler)
			* [`ForecastingPipelineTypes.ExogenousFlattenEnsembler`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousFlattenEnsembler)
			* [`ForecastingPipelineTypes.ExogenousLocalizedFlattenEnsembler`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousLocalizedFlattenEnsembler)
			* [`ForecastingPipelineTypes.ExogenousMT2RForecaster`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousMT2RForecaster)
			* [`ForecastingPipelineTypes.ExogenousRandomForestRegressor`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousRandomForestRegressor)
			* [`ForecastingPipelineTypes.ExogenousSVM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.ExogenousSVM)
			* [`ForecastingPipelineTypes.FlattenEnsembler`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.FlattenEnsembler)
			* [`ForecastingPipelineTypes.HoltWinterAdditive`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.HoltWinterAdditive)
			* [`ForecastingPipelineTypes.HoltWinterMultiplicative`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.HoltWinterMultiplicative)
			* [`ForecastingPipelineTypes.LocalizedFlattenEnsembler`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.LocalizedFlattenEnsembler)
			* [`ForecastingPipelineTypes.MT2RForecaster`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.MT2RForecaster)
			* [`ForecastingPipelineTypes.RandomForestRegressor`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.RandomForestRegressor)
			* [`ForecastingPipelineTypes.SVM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.SVM)
			* [`ForecastingPipelineTypes.get_exogenous()`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.get_exogenous)
			* [`ForecastingPipelineTypes.get_non_exogenous()`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ForecastingPipelineTypes.get_non_exogenous)
		- [`ImputationStrategy`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy)
			* [`ImputationStrategy.BEST_OF_DEFAULT_IMPUTERS`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.BEST_OF_DEFAULT_IMPUTERS)
			* [`ImputationStrategy.CUBIC`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.CUBIC)
			* [`ImputationStrategy.FLATTEN_ITERATIVE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.FLATTEN_ITERATIVE)
			* [`ImputationStrategy.LINEAR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.LINEAR)
			* [`ImputationStrategy.MEAN`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.MEAN)
			* [`ImputationStrategy.MEDIAN`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.MEDIAN)
			* [`ImputationStrategy.MOST_FREQUENT`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.MOST_FREQUENT)
			* [`ImputationStrategy.NEXT`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.NEXT)
			* [`ImputationStrategy.NO_IMPUTATION`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.NO_IMPUTATION)
			* [`ImputationStrategy.PREVIOUS`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.PREVIOUS)
			* [`ImputationStrategy.VALUE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.ImputationStrategy.VALUE)
		- [`Metrics`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics)
			* [`Metrics.ACCURACY_AND_DISPARATE_IMPACT_SCORE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.ACCURACY_AND_DISPARATE_IMPACT_SCORE)
			* [`Metrics.ACCURACY_SCORE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.ACCURACY_SCORE)
			* [`Metrics.AVERAGE_PRECISION_SCORE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.AVERAGE_PRECISION_SCORE)
			* [`Metrics.EXPLAINED_VARIANCE_SCORE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.EXPLAINED_VARIANCE_SCORE)
			* [`Metrics.F1_SCORE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.F1_SCORE)
			* [`Metrics.F1_SCORE_MACRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.F1_SCORE_MACRO)
			* [`Metrics.F1_SCORE_MICRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.F1_SCORE_MICRO)
			* [`Metrics.F1_SCORE_WEIGHTED`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.F1_SCORE_WEIGHTED)
			* [`Metrics.LOG_LOSS`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.LOG_LOSS)
			* [`Metrics.MEAN_ABSOLUTE_ERROR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.MEAN_ABSOLUTE_ERROR)
			* [`Metrics.MEAN_SQUARED_ERROR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.MEAN_SQUARED_ERROR)
			* [`Metrics.MEAN_SQUARED_LOG_ERROR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.MEAN_SQUARED_LOG_ERROR)
			* [`Metrics.MEDIAN_ABSOLUTE_ERROR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.MEDIAN_ABSOLUTE_ERROR)
			* [`Metrics.PRECISION_SCORE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.PRECISION_SCORE)
			* [`Metrics.PRECISION_SCORE_MACRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.PRECISION_SCORE_MACRO)
			* [`Metrics.PRECISION_SCORE_MICRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.PRECISION_SCORE_MICRO)
			* [`Metrics.PRECISION_SCORE_WEIGHTED`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.PRECISION_SCORE_WEIGHTED)
			* [`Metrics.R2_AND_DISPARATE_IMPACT_SCORE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.R2_AND_DISPARATE_IMPACT_SCORE)
			* [`Metrics.R2_SCORE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.R2_SCORE)
			* [`Metrics.RECALL_SCORE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.RECALL_SCORE)
			* [`Metrics.RECALL_SCORE_MACRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.RECALL_SCORE_MACRO)
			* [`Metrics.RECALL_SCORE_MICRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.RECALL_SCORE_MICRO)
			* [`Metrics.RECALL_SCORE_WEIGHTED`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.RECALL_SCORE_WEIGHTED)
			* [`Metrics.ROC_AUC_SCORE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.ROC_AUC_SCORE)
			* [`Metrics.ROOT_MEAN_SQUARED_ERROR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.ROOT_MEAN_SQUARED_ERROR)
			* [`Metrics.ROOT_MEAN_SQUARED_LOG_ERROR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Metrics.ROOT_MEAN_SQUARED_LOG_ERROR)
		- [`MetricsToDirections`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections)
			* [`MetricsToDirections.ACCURACY`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.ACCURACY)
			* [`MetricsToDirections.AVERAGE_PRECISION`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.AVERAGE_PRECISION)
			* [`MetricsToDirections.EXPLAINED_VARIANCE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.EXPLAINED_VARIANCE)
			* [`MetricsToDirections.F1`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.F1)
			* [`MetricsToDirections.F1_MACRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.F1_MACRO)
			* [`MetricsToDirections.F1_MICRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.F1_MICRO)
			* [`MetricsToDirections.F1_WEIGHTED`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.F1_WEIGHTED)
			* [`MetricsToDirections.NEG_LOG_LOSS`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_LOG_LOSS)
			* [`MetricsToDirections.NEG_MEAN_ABSOLUTE_ERROR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_MEAN_ABSOLUTE_ERROR)
			* [`MetricsToDirections.NEG_MEAN_SQUARED_ERROR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_MEAN_SQUARED_ERROR)
			* [`MetricsToDirections.NEG_MEAN_SQUARED_LOG_ERROR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_MEAN_SQUARED_LOG_ERROR)
			* [`MetricsToDirections.NEG_MEDIAN_ABSOLUTE_ERROR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_MEDIAN_ABSOLUTE_ERROR)
			* [`MetricsToDirections.NEG_ROOT_MEAN_SQUARED_ERROR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_ROOT_MEAN_SQUARED_ERROR)
			* [`MetricsToDirections.NEG_ROOT_MEAN_SQUARED_LOG_ERROR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NEG_ROOT_MEAN_SQUARED_LOG_ERROR)
			* [`MetricsToDirections.NORMALIZED_GINI_COEFFICIENT`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.NORMALIZED_GINI_COEFFICIENT)
			* [`MetricsToDirections.PRECISION`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.PRECISION)
			* [`MetricsToDirections.PRECISION_MACRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.PRECISION_MACRO)
			* [`MetricsToDirections.PRECISION_MICRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.PRECISION_MICRO)
			* [`MetricsToDirections.PRECISION_WEIGHTED`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.PRECISION_WEIGHTED)
			* [`MetricsToDirections.R2`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.R2)
			* [`MetricsToDirections.RECALL`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.RECALL)
			* [`MetricsToDirections.RECALL_MACRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.RECALL_MACRO)
			* [`MetricsToDirections.RECALL_MICRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.RECALL_MICRO)
			* [`MetricsToDirections.RECALL_WEIGHTED`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.RECALL_WEIGHTED)
			* [`MetricsToDirections.ROC_AUC`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.MetricsToDirections.ROC_AUC)
		- [`PipelineTypes`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PipelineTypes)
			* [`PipelineTypes.LALE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PipelineTypes.LALE)
			* [`PipelineTypes.SKLEARN`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PipelineTypes.SKLEARN)
		- [`PositiveLabelClass`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass)
			* [`PositiveLabelClass.AVERAGE_PRECISION_SCORE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.AVERAGE_PRECISION_SCORE)
			* [`PositiveLabelClass.F1_SCORE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.F1_SCORE)
			* [`PositiveLabelClass.F1_SCORE_MACRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.F1_SCORE_MACRO)
			* [`PositiveLabelClass.F1_SCORE_MICRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.F1_SCORE_MICRO)
			* [`PositiveLabelClass.F1_SCORE_WEIGHTED`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.F1_SCORE_WEIGHTED)
			* [`PositiveLabelClass.PRECISION_SCORE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.PRECISION_SCORE)
			* [`PositiveLabelClass.PRECISION_SCORE_MACRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.PRECISION_SCORE_MACRO)
			* [`PositiveLabelClass.PRECISION_SCORE_MICRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.PRECISION_SCORE_MICRO)
			* [`PositiveLabelClass.PRECISION_SCORE_WEIGHTED`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.PRECISION_SCORE_WEIGHTED)
			* [`PositiveLabelClass.RECALL_SCORE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.RECALL_SCORE)
			* [`PositiveLabelClass.RECALL_SCORE_MACRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.RECALL_SCORE_MACRO)
			* [`PositiveLabelClass.RECALL_SCORE_MICRO`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.RECALL_SCORE_MICRO)
			* [`PositiveLabelClass.RECALL_SCORE_WEIGHTED`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PositiveLabelClass.RECALL_SCORE_WEIGHTED)
		- [`PredictionType`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PredictionType)
			* [`PredictionType.BINARY`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PredictionType.BINARY)
			* [`PredictionType.CLASSIFICATION`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PredictionType.CLASSIFICATION)
			* [`PredictionType.FORECASTING`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PredictionType.FORECASTING)
			* [`PredictionType.MULTICLASS`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PredictionType.MULTICLASS)
			* [`PredictionType.REGRESSION`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PredictionType.REGRESSION)
			* [`PredictionType.TIMESERIES_ANOMALY_PREDICTION`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.PredictionType.TIMESERIES_ANOMALY_PREDICTION)
		- [`RegressionAlgorithms`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms)
			* [`RegressionAlgorithms.DT`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.DT)
			* [`RegressionAlgorithms.EX_TREES`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.EX_TREES)
			* [`RegressionAlgorithms.GB`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.GB)
			* [`RegressionAlgorithms.LGBM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.LGBM)
			* [`RegressionAlgorithms.LR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.LR)
			* [`RegressionAlgorithms.RF`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.RF)
			* [`RegressionAlgorithms.RIDGE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.RIDGE)
			* [`RegressionAlgorithms.SnapBM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.SnapBM)
			* [`RegressionAlgorithms.SnapDT`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.SnapDT)
			* [`RegressionAlgorithms.SnapRF`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.SnapRF)
			* [`RegressionAlgorithms.XGB`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithms.XGB)
		- [`RegressionAlgorithmsCP4D`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D)
			* [`RegressionAlgorithmsCP4D.DT`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.DT)
			* [`RegressionAlgorithmsCP4D.EX_TREES`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.EX_TREES)
			* [`RegressionAlgorithmsCP4D.GB`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.GB)
			* [`RegressionAlgorithmsCP4D.LGBM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.LGBM)
			* [`RegressionAlgorithmsCP4D.LR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.LR)
			* [`RegressionAlgorithmsCP4D.RF`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.RF)
			* [`RegressionAlgorithmsCP4D.RIDGE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.RIDGE)
			* [`RegressionAlgorithmsCP4D.SnapBM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.SnapBM)
			* [`RegressionAlgorithmsCP4D.SnapDT`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.SnapDT)
			* [`RegressionAlgorithmsCP4D.SnapRF`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.SnapRF)
			* [`RegressionAlgorithmsCP4D.XGB`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RegressionAlgorithmsCP4D.XGB)
		- [`RunStateTypes`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RunStateTypes)
			* [`RunStateTypes.COMPLETED`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RunStateTypes.COMPLETED)
			* [`RunStateTypes.FAILED`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.RunStateTypes.FAILED)
		- [`SamplingTypes`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.SamplingTypes)
			* [`SamplingTypes.FIRST_VALUES`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.SamplingTypes.FIRST_VALUES)
			* [`SamplingTypes.LAST_VALUES`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.SamplingTypes.LAST_VALUES)
			* [`SamplingTypes.RANDOM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.SamplingTypes.RANDOM)
			* [`SamplingTypes.STRATIFIED`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.SamplingTypes.STRATIFIED)
		- [`TShirtSize`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TShirtSize)
			* [`TShirtSize.L`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TShirtSize.L)
			* [`TShirtSize.M`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TShirtSize.M)
			* [`TShirtSize.S`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TShirtSize.S)
			* [`TShirtSize.XL`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TShirtSize.XL)
		- [`TimeseriesAnomalyPredictionAlgorithms`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionAlgorithms)
			* [`TimeseriesAnomalyPredictionAlgorithms.Forecasting`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionAlgorithms.Forecasting)
			* [`TimeseriesAnomalyPredictionAlgorithms.Relationship`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionAlgorithms.Relationship)
			* [`TimeseriesAnomalyPredictionAlgorithms.Window`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionAlgorithms.Window)
		- [`TimeseriesAnomalyPredictionPipelineTypes`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes)
			* [`TimeseriesAnomalyPredictionPipelineTypes.PointwiseBoundedBATS`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.PointwiseBoundedBATS)
			* [`TimeseriesAnomalyPredictionPipelineTypes.PointwiseBoundedBATSForceUpdate`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.PointwiseBoundedBATSForceUpdate)
			* [`TimeseriesAnomalyPredictionPipelineTypes.PointwiseBoundedHoltWintersAdditive`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.PointwiseBoundedHoltWintersAdditive)
			* [`TimeseriesAnomalyPredictionPipelineTypes.WindowLOF`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.WindowLOF)
			* [`TimeseriesAnomalyPredictionPipelineTypes.WindowNN`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.WindowNN)
			* [`TimeseriesAnomalyPredictionPipelineTypes.WindowPCA`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.TimeseriesAnomalyPredictionPipelineTypes.WindowPCA)
		- [`Transformers`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers)
			* [`Transformers.ABS`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.ABS)
			* [`Transformers.CBRT`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.CBRT)
			* [`Transformers.COS`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.COS)
			* [`Transformers.CUBE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.CUBE)
			* [`Transformers.DIFF`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.DIFF)
			* [`Transformers.DIVIDE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.DIVIDE)
			* [`Transformers.FEATUREAGGLOMERATION`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.FEATUREAGGLOMERATION)
			* [`Transformers.ISOFORESTANOMALY`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.ISOFORESTANOMALY)
			* [`Transformers.LOG`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.LOG)
			* [`Transformers.MAX`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.MAX)
			* [`Transformers.MINMAXSCALER`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.MINMAXSCALER)
			* [`Transformers.NXOR`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.NXOR)
			* [`Transformers.PCA`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.PCA)
			* [`Transformers.PRODUCT`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.PRODUCT)
			* [`Transformers.ROUND`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.ROUND)
			* [`Transformers.SIGMOID`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.SIGMOID)
			* [`Transformers.SIN`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.SIN)
			* [`Transformers.SQRT`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.SQRT)
			* [`Transformers.SQUARE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.SQUARE)
			* [`Transformers.STDSCALER`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.STDSCALER)
			* [`Transformers.SUM`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.SUM)
			* [`Transformers.TAN`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.Transformers.TAN)
		- [`VisualizationTypes`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.VisualizationTypes)
			* [`VisualizationTypes.INPLACE`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.VisualizationTypes.INPLACE)
			* [`VisualizationTypes.PDF`](core_api.html#ibm_watsonx_ai.utils.autoai.enums.VisualizationTypes.PDF)
* [Federated Learning](federated_learning.html)
	+ [Aggregation](federated_learning.html#aggregation)
		- [Configure and start aggregation](federated_learning.html#configure-and-start-aggregation)
	+ [Local training](federated_learning.html#local-training)
		- [Configure and start local training](federated_learning.html#configure-and-start-local-training)
			* [`RemoteTrainingSystem`](federated_learning.html#remote_training_system.RemoteTrainingSystem)
				+ [`RemoteTrainingSystem.create_party()`](federated_learning.html#remote_training_system.RemoteTrainingSystem.create_party)
				+ [`RemoteTrainingSystem.create_revision()`](federated_learning.html#remote_training_system.RemoteTrainingSystem.create_revision)
				+ [`RemoteTrainingSystem.delete()`](federated_learning.html#remote_training_system.RemoteTrainingSystem.delete)
				+ [`RemoteTrainingSystem.get_details()`](federated_learning.html#remote_training_system.RemoteTrainingSystem.get_details)
				+ [`RemoteTrainingSystem.get_id()`](federated_learning.html#remote_training_system.RemoteTrainingSystem.get_id)
				+ [`RemoteTrainingSystem.get_revision_details()`](federated_learning.html#remote_training_system.RemoteTrainingSystem.get_revision_details)
				+ [`RemoteTrainingSystem.list()`](federated_learning.html#remote_training_system.RemoteTrainingSystem.list)
				+ [`RemoteTrainingSystem.list_revisions()`](federated_learning.html#remote_training_system.RemoteTrainingSystem.list_revisions)
				+ [`RemoteTrainingSystem.store()`](federated_learning.html#remote_training_system.RemoteTrainingSystem.store)
				+ [`RemoteTrainingSystem.update()`](federated_learning.html#remote_training_system.RemoteTrainingSystem.update)
			* [`Party`](federated_learning.html#party_wrapper.Party)
				+ [`Party.cancel()`](federated_learning.html#party_wrapper.Party.cancel)
				+ [`Party.get_round()`](federated_learning.html#party_wrapper.Party.get_round)
				+ [`Party.is_running()`](federated_learning.html#party_wrapper.Party.is_running)
				+ [`Party.monitor_logs()`](federated_learning.html#party_wrapper.Party.monitor_logs)
				+ [`Party.monitor_metrics()`](federated_learning.html#party_wrapper.Party.monitor_metrics)
				+ [`Party.run()`](federated_learning.html#party_wrapper.Party.run)
* [DataConnection](dataconnection.html)
	+ [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [IBM Cloud - DataConnection Initialization](autoai_working_with_dataconnection.html#ibm-cloud-dataconnection-initialization)
			* [Connection Asset](autoai_working_with_dataconnection.html#connection-asset)
			* [Data Asset](autoai_working_with_dataconnection.html#data-asset)
			* [Container](autoai_working_with_dataconnection.html#container)
		- [IBM watsonx.ai software - DataConnection Initialization](autoai_working_with_dataconnection.html#ibm-watsonx-ai-software-dataconnection-initialization)
			* [Connection Asset - DatabaseLocation](autoai_working_with_dataconnection.html#connection-asset-databaselocation)
			* [Connection Asset - S3Location](autoai_working_with_dataconnection.html#connection-asset-s3location)
			* [Connection Asset - NFSLocation](autoai_working_with_dataconnection.html#connection-asset-nfslocation)
			* [Data Asset](autoai_working_with_dataconnection.html#id1)
			* [FSLocation](autoai_working_with_dataconnection.html#fslocation)
		- [Batch DataConnection](autoai_working_with_dataconnection.html#batch-dataconnection)
		- [Upload your training dataset](autoai_working_with_dataconnection.html#upload-your-training-dataset)
		- [Download your training dataset](autoai_working_with_dataconnection.html#download-your-training-dataset)
	+ [DataConnection Modules](dataconnection_modules.html)
		- [DataConnection](dataconnection_modules.html#dataconnection)
			* [`DataConnection`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DataConnection)
				+ [`DataConnection.from_studio()`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DataConnection.from_studio)
				+ [`DataConnection.read()`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DataConnection.read)
				+ [`DataConnection.set_client()`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DataConnection.set_client)
				+ [`DataConnection.write()`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DataConnection.write)
		- [S3Location](dataconnection_modules.html#s3location)
			* [`S3Location`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.S3Location)
				+ [`S3Location.get_location()`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.S3Location.get_location)
		- [CloudAssetLocation](dataconnection_modules.html#cloudassetlocation)
			* [`CloudAssetLocation`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.CloudAssetLocation)
		- [DeploymentOutputAssetLocation](dataconnection_modules.html#deploymentoutputassetlocation)
			* [`DeploymentOutputAssetLocation`](dataconnection_modules.html#ibm_watsonx_ai.helpers.connections.connections.DeploymentOutputAssetLocation)
* [AutoAI](autoai.html)
	+ [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [Configure optimizer with one data source](autoai_working_with_class_and_optimizer.html#configure-optimizer-with-one-data-source)
		- [Configure optimizer for time series forecasting](autoai_working_with_class_and_optimizer.html#configure-optimizer-for-time-series-forecasting)
		- [Configure optimizer for time series forecasting with supporting features](autoai_working_with_class_and_optimizer.html#configure-optimizer-for-time-series-forecasting-with-supporting-features)
		- [Get configuration parameters](autoai_working_with_class_and_optimizer.html#get-configuration-parameters)
		- [Fit optimizer](autoai_working_with_class_and_optimizer.html#fit-optimizer)
		- [Get the run status and run details](autoai_working_with_class_and_optimizer.html#get-the-run-status-and-run-details)
		- [Get data connections](autoai_working_with_class_and_optimizer.html#get-data-connections)
		- [Pipeline summary](autoai_working_with_class_and_optimizer.html#pipeline-summary)
		- [Get pipeline details](autoai_working_with_class_and_optimizer.html#get-pipeline-details)
		- [Get pipeline](autoai_working_with_class_and_optimizer.html#get-pipeline)
		- [Working with deployments](autoai_working_with_class_and_optimizer.html#working-with-deployments)
		- [Web Service](autoai_working_with_class_and_optimizer.html#web-service)
		- [Batch](autoai_working_with_class_and_optimizer.html#batch)
	+ [AutoAI experiment](autoai_experiment.html)
		- [AutoAI](autoai_experiment.html#autoai)
			* [`AutoAI`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI)
				+ [`AutoAI.ClassificationAlgorithms`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms)
					- [`AutoAI.ClassificationAlgorithms.DT`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.DT)
					- [`AutoAI.ClassificationAlgorithms.EX_TREES`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.EX_TREES)
					- [`AutoAI.ClassificationAlgorithms.GB`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.GB)
					- [`AutoAI.ClassificationAlgorithms.LGBM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.LGBM)
					- [`AutoAI.ClassificationAlgorithms.LR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.LR)
					- [`AutoAI.ClassificationAlgorithms.RF`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.RF)
					- [`AutoAI.ClassificationAlgorithms.SnapBM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapBM)
					- [`AutoAI.ClassificationAlgorithms.SnapDT`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapDT)
					- [`AutoAI.ClassificationAlgorithms.SnapLR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapLR)
					- [`AutoAI.ClassificationAlgorithms.SnapRF`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapRF)
					- [`AutoAI.ClassificationAlgorithms.SnapSVM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.SnapSVM)
					- [`AutoAI.ClassificationAlgorithms.XGB`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ClassificationAlgorithms.XGB)
				+ [`AutoAI.DataConnectionTypes`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes)
					- [`AutoAI.DataConnectionTypes.CA`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.CA)
					- [`AutoAI.DataConnectionTypes.CN`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.CN)
					- [`AutoAI.DataConnectionTypes.DS`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.DS)
					- [`AutoAI.DataConnectionTypes.FS`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.FS)
					- [`AutoAI.DataConnectionTypes.S3`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.DataConnectionTypes.S3)
				+ [`AutoAI.ForecastingAlgorithms`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms)
					- [`AutoAI.ForecastingAlgorithms.ARIMA`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.ARIMA)
					- [`AutoAI.ForecastingAlgorithms.BATS`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.BATS)
					- [`AutoAI.ForecastingAlgorithms.ENSEMBLER`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.ENSEMBLER)
					- [`AutoAI.ForecastingAlgorithms.HW`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.HW)
					- [`AutoAI.ForecastingAlgorithms.LR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.LR)
					- [`AutoAI.ForecastingAlgorithms.RF`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.RF)
					- [`AutoAI.ForecastingAlgorithms.SVM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.ForecastingAlgorithms.SVM)
				+ [`AutoAI.Metrics`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics)
					- [`AutoAI.Metrics.ACCURACY_AND_DISPARATE_IMPACT_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ACCURACY_AND_DISPARATE_IMPACT_SCORE)
					- [`AutoAI.Metrics.ACCURACY_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ACCURACY_SCORE)
					- [`AutoAI.Metrics.AVERAGE_PRECISION_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.AVERAGE_PRECISION_SCORE)
					- [`AutoAI.Metrics.EXPLAINED_VARIANCE_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.EXPLAINED_VARIANCE_SCORE)
					- [`AutoAI.Metrics.F1_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE)
					- [`AutoAI.Metrics.F1_SCORE_MACRO`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE_MACRO)
					- [`AutoAI.Metrics.F1_SCORE_MICRO`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE_MICRO)
					- [`AutoAI.Metrics.F1_SCORE_WEIGHTED`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.F1_SCORE_WEIGHTED)
					- [`AutoAI.Metrics.LOG_LOSS`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.LOG_LOSS)
					- [`AutoAI.Metrics.MEAN_ABSOLUTE_ERROR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEAN_ABSOLUTE_ERROR)
					- [`AutoAI.Metrics.MEAN_SQUARED_ERROR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEAN_SQUARED_ERROR)
					- [`AutoAI.Metrics.MEAN_SQUARED_LOG_ERROR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEAN_SQUARED_LOG_ERROR)
					- [`AutoAI.Metrics.MEDIAN_ABSOLUTE_ERROR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.MEDIAN_ABSOLUTE_ERROR)
					- [`AutoAI.Metrics.PRECISION_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE)
					- [`AutoAI.Metrics.PRECISION_SCORE_MACRO`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE_MACRO)
					- [`AutoAI.Metrics.PRECISION_SCORE_MICRO`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE_MICRO)
					- [`AutoAI.Metrics.PRECISION_SCORE_WEIGHTED`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.PRECISION_SCORE_WEIGHTED)
					- [`AutoAI.Metrics.R2_AND_DISPARATE_IMPACT_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.R2_AND_DISPARATE_IMPACT_SCORE)
					- [`AutoAI.Metrics.R2_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.R2_SCORE)
					- [`AutoAI.Metrics.RECALL_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE)
					- [`AutoAI.Metrics.RECALL_SCORE_MACRO`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE_MACRO)
					- [`AutoAI.Metrics.RECALL_SCORE_MICRO`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE_MICRO)
					- [`AutoAI.Metrics.RECALL_SCORE_WEIGHTED`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.RECALL_SCORE_WEIGHTED)
					- [`AutoAI.Metrics.ROC_AUC_SCORE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ROC_AUC_SCORE)
					- [`AutoAI.Metrics.ROOT_MEAN_SQUARED_ERROR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ROOT_MEAN_SQUARED_ERROR)
					- [`AutoAI.Metrics.ROOT_MEAN_SQUARED_LOG_ERROR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Metrics.ROOT_MEAN_SQUARED_LOG_ERROR)
				+ [`AutoAI.PipelineTypes`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PipelineTypes)
					- [`AutoAI.PipelineTypes.LALE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PipelineTypes.LALE)
					- [`AutoAI.PipelineTypes.SKLEARN`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PipelineTypes.SKLEARN)
				+ [`AutoAI.PredictionType`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType)
					- [`AutoAI.PredictionType.BINARY`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.BINARY)
					- [`AutoAI.PredictionType.CLASSIFICATION`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.CLASSIFICATION)
					- [`AutoAI.PredictionType.FORECASTING`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.FORECASTING)
					- [`AutoAI.PredictionType.MULTICLASS`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.MULTICLASS)
					- [`AutoAI.PredictionType.REGRESSION`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.REGRESSION)
					- [`AutoAI.PredictionType.TIMESERIES_ANOMALY_PREDICTION`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.PredictionType.TIMESERIES_ANOMALY_PREDICTION)
				+ [`AutoAI.RegressionAlgorithms`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms)
					- [`AutoAI.RegressionAlgorithms.DT`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.DT)
					- [`AutoAI.RegressionAlgorithms.EX_TREES`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.EX_TREES)
					- [`AutoAI.RegressionAlgorithms.GB`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.GB)
					- [`AutoAI.RegressionAlgorithms.LGBM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.LGBM)
					- [`AutoAI.RegressionAlgorithms.LR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.LR)
					- [`AutoAI.RegressionAlgorithms.RF`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.RF)
					- [`AutoAI.RegressionAlgorithms.RIDGE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.RIDGE)
					- [`AutoAI.RegressionAlgorithms.SnapBM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.SnapBM)
					- [`AutoAI.RegressionAlgorithms.SnapDT`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.SnapDT)
					- [`AutoAI.RegressionAlgorithms.SnapRF`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.SnapRF)
					- [`AutoAI.RegressionAlgorithms.XGB`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.RegressionAlgorithms.XGB)
				+ [`AutoAI.SamplingTypes`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes)
					- [`AutoAI.SamplingTypes.FIRST_VALUES`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.FIRST_VALUES)
					- [`AutoAI.SamplingTypes.LAST_VALUES`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.LAST_VALUES)
					- [`AutoAI.SamplingTypes.RANDOM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.RANDOM)
					- [`AutoAI.SamplingTypes.STRATIFIED`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.SamplingTypes.STRATIFIED)
				+ [`AutoAI.TShirtSize`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize)
					- [`AutoAI.TShirtSize.L`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.L)
					- [`AutoAI.TShirtSize.M`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.M)
					- [`AutoAI.TShirtSize.S`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.S)
					- [`AutoAI.TShirtSize.XL`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.TShirtSize.XL)
				+ [`AutoAI.Transformers`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers)
					- [`AutoAI.Transformers.ABS`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.ABS)
					- [`AutoAI.Transformers.CBRT`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.CBRT)
					- [`AutoAI.Transformers.COS`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.COS)
					- [`AutoAI.Transformers.CUBE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.CUBE)
					- [`AutoAI.Transformers.DIFF`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.DIFF)
					- [`AutoAI.Transformers.DIVIDE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.DIVIDE)
					- [`AutoAI.Transformers.FEATUREAGGLOMERATION`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.FEATUREAGGLOMERATION)
					- [`AutoAI.Transformers.ISOFORESTANOMALY`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.ISOFORESTANOMALY)
					- [`AutoAI.Transformers.LOG`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.LOG)
					- [`AutoAI.Transformers.MAX`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.MAX)
					- [`AutoAI.Transformers.MINMAXSCALER`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.MINMAXSCALER)
					- [`AutoAI.Transformers.NXOR`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.NXOR)
					- [`AutoAI.Transformers.PCA`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.PCA)
					- [`AutoAI.Transformers.PRODUCT`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.PRODUCT)
					- [`AutoAI.Transformers.ROUND`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.ROUND)
					- [`AutoAI.Transformers.SIGMOID`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SIGMOID)
					- [`AutoAI.Transformers.SIN`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SIN)
					- [`AutoAI.Transformers.SQRT`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SQRT)
					- [`AutoAI.Transformers.SQUARE`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SQUARE)
					- [`AutoAI.Transformers.STDSCALER`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.STDSCALER)
					- [`AutoAI.Transformers.SUM`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.SUM)
					- [`AutoAI.Transformers.TAN`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.Transformers.TAN)
				+ [`AutoAI.optimizer()`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.optimizer)
				+ [`AutoAI.runs()`](autoai_experiment.html#ibm_watsonx_ai.experiment.autoai.autoai.AutoAI.runs)
	+ [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
		- [Web Service](autoai_deployment_modules.html#web-service)
			* [`WebService`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.WebService)
				+ [`WebService.create()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.WebService.create)
				+ [`WebService.delete()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.WebService.delete)
				+ [`WebService.get()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.WebService.get)
				+ [`WebService.get_params()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.WebService.get_params)
				+ [`WebService.list()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.WebService.list)
				+ [`WebService.score()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.WebService.score)
		- [Batch](autoai_deployment_modules.html#batch)
			* [`Batch`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch)
				+ [`Batch.create()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.create)
				+ [`Batch.delete()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.delete)
				+ [`Batch.get()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.get)
				+ [`Batch.get_job_id()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.get_job_id)
				+ [`Batch.get_job_params()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.get_job_params)
				+ [`Batch.get_job_result()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.get_job_result)
				+ [`Batch.get_job_status()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.get_job_status)
				+ [`Batch.get_params()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.get_params)
				+ [`Batch.list()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.list)
				+ [`Batch.list_jobs()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.list_jobs)
				+ [`Batch.rerun_job()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.rerun_job)
				+ [`Batch.run_job()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.run_job)
				+ [`Batch.score()`](autoai_deployment_modules.html#ibm_watsonx_ai.deployment.Batch.score)
* [Foundation Models](foundation_models.html)
	+ [Modules](foundation_models.html#modules)
		- [Embeddings](fm_embeddings.html)
			* [Embeddings](fm_embeddings.html#id1)
				+ [`Embeddings`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.Embeddings)
					- [`Embeddings.embed_documents()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.embed_documents)
					- [`Embeddings.embed_query()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.embed_query)
					- [`Embeddings.generate()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.generate)
					- [`Embeddings.to_dict()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.Embeddings.to_dict)
			* [BaseEmbeddings](fm_embeddings.html#baseembeddings)
				+ [`BaseEmbeddings`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings)
					- [`BaseEmbeddings.embed_documents()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.embed_documents)
					- [`BaseEmbeddings.embed_query()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.embed_query)
					- [`BaseEmbeddings.from_dict()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.from_dict)
					- [`BaseEmbeddings.to_dict()`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.embeddings.base_embeddings.BaseEmbeddings.to_dict)
			* [Enums](fm_embeddings.html#enums)
				+ [`EmbedTextParamsMetaNames`](fm_embeddings.html#metanames.EmbedTextParamsMetaNames)
				+ [`EmbeddingModels`](fm_embeddings.html#EmbeddingModels)
				+ [`EmbeddingTypes`](fm_embeddings.html#ibm_watsonx_ai.foundation_models.utils.enums.EmbeddingTypes)
		- [Models](fm_models.html)
			* [Modules](fm_models.html#modules)
				+ [Model](fm_model.html)
					- [`Model`](fm_model.html#ibm_watsonx_ai.foundation_models.Model)
						* [`Model.generate()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.generate)
						* [`Model.generate_text()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.generate_text)
						* [`Model.generate_text_stream()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.generate_text_stream)
						* [`Model.get_details()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.get_details)
						* [`Model.to_langchain()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.to_langchain)
						* [`Model.tokenize()`](fm_model.html#ibm_watsonx_ai.foundation_models.Model.tokenize)
					- [Enums](fm_model.html#enums)
						* [`GenTextParamsMetaNames`](fm_model.html#metanames.GenTextParamsMetaNames)
						* [`GenTextReturnOptMetaNames`](fm_model.html#metanames.GenTextReturnOptMetaNames)
						* [`DecodingMethods`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods)
							+ [`DecodingMethods.GREEDY`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods.GREEDY)
							+ [`DecodingMethods.SAMPLE`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.DecodingMethods.SAMPLE)
						* [`TextModels`](fm_model.html#TextModels)
						* [`ModelTypes`](fm_model.html#ibm_watsonx_ai.foundation_models.utils.enums.ModelTypes)
				+ [ModelInference](fm_model_inference.html)
					- [`ModelInference`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference)
						* [`ModelInference.generate()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate)
						* [`ModelInference.generate_text()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate_text)
						* [`ModelInference.generate_text_stream()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.generate_text_stream)
						* [`ModelInference.get_details()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.get_details)
						* [`ModelInference.get_identifying_params()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.get_identifying_params)
						* [`ModelInference.to_langchain()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.to_langchain)
						* [`ModelInference.tokenize()`](fm_model_inference.html#ibm_watsonx_ai.foundation_models.inference.ModelInference.tokenize)
				+ [`ModelInference` for Deployments](fm_deployments.html)
					- [Infer text with deployments](fm_deployments.html#infer-text-with-deployments)
					- [Creating `ModelInference` instance](fm_deployments.html#creating-modelinference-instance)
					- [Generate methods](fm_deployments.html#generate-methods)
		- [Prompt Tuning](prompt_tuner.html)
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)
				+ [Tune Experiment run](pt_tune_experiment_run.html)
					- [Configure PromptTuner](pt_tune_experiment_run.html#configure-prompttuner)
					- [Get configuration parameters](pt_tune_experiment_run.html#get-configuration-parameters)
					- [Run prompt tuning](pt_tune_experiment_run.html#run-prompt-tuning)
					- [Get run status, get run details](pt_tune_experiment_run.html#get-run-status-get-run-details)
					- [Get data connections](pt_tune_experiment_run.html#get-data-connections)
					- [Summary](pt_tune_experiment_run.html#summary)
					- [Plot learning curves](pt_tune_experiment_run.html#plot-learning-curves)
					- [Get model identifier](pt_tune_experiment_run.html#get-model-identifier)
				+ [Tuned Model Inference](pt_model_inference.html)
					- [Working with deployments](pt_model_inference.html#working-with-deployments)
					- [Creating `ModelInference` instance](pt_model_inference.html#creating-modelinference-instance)
					- [Importing data](pt_model_inference.html#importing-data)
					- [Analyzing satisfaction](pt_model_inference.html#analyzing-satisfaction)
					- [Generate methods](pt_model_inference.html#generate-methods)
			* [Tune Experiment](tune_experiment.html)
				+ [TuneExperiment](tune_experiment.html#tuneexperiment)
					- [`TuneExperiment`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneExperiment)
						* [`TuneExperiment.prompt_tuner()`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneExperiment.prompt_tuner)
						* [`TuneExperiment.runs()`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneExperiment.runs)
				+ [Tune Runs](tune_experiment.html#tune-runs)
					- [`TuneRuns`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneRuns)
						* [`TuneRuns.get_run_details()`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneRuns.get_run_details)
						* [`TuneRuns.get_tuner()`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneRuns.get_tuner)
						* [`TuneRuns.list()`](tune_experiment.html#ibm_watsonx_ai.experiment.fm_tune.TuneRuns.list)
				+ [Prompt Tuner](tune_experiment.html#prompt-tuner)
					- [`PromptTuner`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner)
						* [`PromptTuner.cancel_run()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.cancel_run)
						* [`PromptTuner.get_data_connections()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.get_data_connections)
						* [`PromptTuner.get_model_id()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.get_model_id)
						* [`PromptTuner.get_params()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.get_params)
						* [`PromptTuner.get_run_details()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.get_run_details)
						* [`PromptTuner.get_run_status()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.get_run_status)
						* [`PromptTuner.plot_learning_curve()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.plot_learning_curve)
						* [`PromptTuner.run()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.run)
						* [`PromptTuner.summary()`](tune_experiment.html#ibm_watsonx_ai.foundation_models.PromptTuner.summary)
				+ [Enums](tune_experiment.html#enums)
					- [`PromptTuningTypes`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningTypes)
						* [`PromptTuningTypes.PT`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningTypes.PT)
					- [`PromptTuningInitMethods`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningInitMethods)
						* [`PromptTuningInitMethods.RANDOM`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningInitMethods.RANDOM)
						* [`PromptTuningInitMethods.TEXT`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTuningInitMethods.TEXT)
					- [`TuneExperimentTasks`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks)
						* [`TuneExperimentTasks.CLASSIFICATION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.CLASSIFICATION)
						* [`TuneExperimentTasks.CODE_GENERATION_AND_CONVERSION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.CODE_GENERATION_AND_CONVERSION)
						* [`TuneExperimentTasks.EXTRACTION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.EXTRACTION)
						* [`TuneExperimentTasks.GENERATION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.GENERATION)
						* [`TuneExperimentTasks.QUESTION_ANSWERING`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.QUESTION_ANSWERING)
						* [`TuneExperimentTasks.RETRIEVAL_AUGMENTED_GENERATION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.RETRIEVAL_AUGMENTED_GENERATION)
						* [`TuneExperimentTasks.SUMMARIZATION`](tune_experiment.html#ibm_watsonx_ai.foundation_models.utils.enums.TuneExperimentTasks.SUMMARIZATION)
					- [`PromptTunableModels`](tune_experiment.html#PromptTunableModels)
		- [Prompt Template Manager](prompt_template_manager.html)
			* [`PromptTemplateManager`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager)
				+ [`PromptTemplateManager.delete_prompt()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.delete_prompt)
				+ [`PromptTemplateManager.get_lock()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.get_lock)
				+ [`PromptTemplateManager.list()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.list)
				+ [`PromptTemplateManager.load_prompt()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.load_prompt)
				+ [`PromptTemplateManager.lock()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.lock)
				+ [`PromptTemplateManager.store_prompt()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.store_prompt)
				+ [`PromptTemplateManager.unlock()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.unlock)
				+ [`PromptTemplateManager.update_prompt()`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplateManager.update_prompt)
			* [`PromptTemplate`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.PromptTemplate)
			* [`FreeformPromptTemplate`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.FreeformPromptTemplate)
			* [`DetachedPromptTemplate`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.prompts.DetachedPromptTemplate)
			* [Enums](prompt_template_manager.html#enums)
				+ [`PromptTemplateFormats`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats)
					- [`PromptTemplateFormats.LANGCHAIN`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats.LANGCHAIN)
					- [`PromptTemplateFormats.PROMPTTEMPLATE`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats.PROMPTTEMPLATE)
					- [`PromptTemplateFormats.STRING`](prompt_template_manager.html#ibm_watsonx_ai.foundation_models.utils.enums.PromptTemplateFormats.STRING)
		- [Extensions](fm_extensions.html)
			* [LangChain](fm_extensions.html#langchain)
				+ [`WatsonxLLM`](fm_extensions.html#langchain_ibm.WatsonxLLM)
					- [`WatsonxLLM.apikey`](fm_extensions.html#langchain_ibm.WatsonxLLM.apikey)
					- [`WatsonxLLM.deployment_id`](fm_extensions.html#langchain_ibm.WatsonxLLM.deployment_id)
					- [`WatsonxLLM.get_num_tokens()`](fm_extensions.html#langchain_ibm.WatsonxLLM.get_num_tokens)
					- [`WatsonxLLM.get_token_ids()`](fm_extensions.html#langchain_ibm.WatsonxLLM.get_token_ids)
					- [`WatsonxLLM.instance_id`](fm_extensions.html#langchain_ibm.WatsonxLLM.instance_id)
					- [`WatsonxLLM.model_id`](fm_extensions.html#langchain_ibm.WatsonxLLM.model_id)
					- [`WatsonxLLM.params`](fm_extensions.html#langchain_ibm.WatsonxLLM.params)
					- [`WatsonxLLM.password`](fm_extensions.html#langchain_ibm.WatsonxLLM.password)
					- [`WatsonxLLM.project_id`](fm_extensions.html#langchain_ibm.WatsonxLLM.project_id)
					- [`WatsonxLLM.space_id`](fm_extensions.html#langchain_ibm.WatsonxLLM.space_id)
					- [`WatsonxLLM.streaming`](fm_extensions.html#langchain_ibm.WatsonxLLM.streaming)
					- [`WatsonxLLM.token`](fm_extensions.html#langchain_ibm.WatsonxLLM.token)
					- [`WatsonxLLM.url`](fm_extensions.html#langchain_ibm.WatsonxLLM.url)
					- [`WatsonxLLM.username`](fm_extensions.html#langchain_ibm.WatsonxLLM.username)
					- [`WatsonxLLM.verify`](fm_extensions.html#langchain_ibm.WatsonxLLM.verify)
					- [`WatsonxLLM.version`](fm_extensions.html#langchain_ibm.WatsonxLLM.version)
		- [Helpers](fm_helpers.html)
			* [`FoundationModelsManager`](fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager)
				+ [`FoundationModelsManager.get_custom_model_specs()`](fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_custom_model_specs)
				+ [`FoundationModelsManager.get_embeddings_model_specs()`](fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_embeddings_model_specs)
				+ [`FoundationModelsManager.get_model_lifecycle()`](fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_model_lifecycle)
				+ [`FoundationModelsManager.get_model_specs()`](fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_model_specs)
				+ [`FoundationModelsManager.get_model_specs_with_prompt_tuning_support()`](fm_helpers.html#ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_model_specs_with_prompt_tuning_support)
			* [`get_model_specs()`](fm_helpers.html#ibm_watsonx_ai.foundation_models.get_model_specs)
			* [`get_model_lifecycle()`](fm_helpers.html#ibm_watsonx_ai.foundation_models.get_model_lifecycle)
			* [`get_model_specs_with_prompt_tuning_support()`](fm_helpers.html#ibm_watsonx_ai.foundation_models.get_model_specs_with_prompt_tuning_support)
			* [`get_supported_tasks()`](fm_helpers.html#ibm_watsonx_ai.foundation_models.get_supported_tasks)
		- [Custom models](fm_working_with_custom_models.html)
			* [Initialize APIClient object](fm_working_with_custom_models.html#initialize-apiclient-object)
			* [Listing models specification](fm_working_with_custom_models.html#listing-models-specification)
			* [Storing model in service repository](fm_working_with_custom_models.html#storing-model-in-service-repository)
			* [Defining hardware specification](fm_working_with_custom_models.html#defining-hardware-specification)
			* [Deployment of custom foundation model](fm_working_with_custom_models.html#deployment-of-custom-foundation-model)
			* [Working with deployments](fm_working_with_custom_models.html#working-with-deployments)









[Next

Base](base.html)
[Previous

IBM watsonx.ai software](setup_cpd.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [API](#)
	+ [Prerequisites](#prerequisites)
	+ [Modules](#modules)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/dataconnection_modules.html








DataConnection Modules - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](#)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





DataConnection Modules[¶](#dataconnection-modules "Link to this heading")
=========================================================================



DataConnection[¶](#dataconnection "Link to this heading")
---------------------------------------------------------




*class* ibm\_watsonx\_ai.helpers.connections.connections.DataConnection(*location=None*, *connection=None*, *data\_asset\_id=None*, *connection\_asset\_id=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/helpers/connections/connections.html#DataConnection)[¶](#ibm_watsonx_ai.helpers.connections.connections.DataConnection "Link to this definition")
Bases: `BaseDataConnection`


Data Storage Connection class needed for Service training metadata (input data).



Parameters:
* **connection** (*NFSConnection* *or* *ConnectionAsset**,* *optional*) – connection parameters of specific type
* **location** (*Union**[*[*S3Location*](#ibm_watsonx_ai.helpers.connections.connections.S3Location "ibm_watsonx_ai.helpers.connections.connections.S3Location")*,* *FSLocation**,* *AssetLocation**]*) – required location parameters of specific type
* **data\_asset\_id** (*str**,* *optional*) – data asset ID if DataConnection should be pointing out to data asset






*classmethod* from\_studio(*path*)[[source]](_modules/ibm_watsonx_ai/helpers/connections/connections.html#DataConnection.from_studio)[¶](#ibm_watsonx_ai.helpers.connections.connections.DataConnection.from_studio "Link to this definition")
Create DataConnections from the credentials stored (connected) in Watson Studio. Only for COS.



Parameters:
**path** (*str*) – path in COS bucket to the training dataset



Returns:
list with DataConnection objects



Return type:
list[[DataConnection](#ibm_watsonx_ai.helpers.connections.connections.DataConnection "ibm_watsonx_ai.helpers.connections.connections.DataConnection")]




**Example**



```
data_connections = DataConnection.from_studio(path='iris_dataset.csv')

```





read(*with\_holdout\_split=False*, *csv\_separator=','*, *excel\_sheet=None*, *encoding='utf-8'*, *raw=False*, *binary=False*, *read\_to\_file=None*, *number\_of\_batch\_rows=None*, *sampling\_type=None*, *sample\_size\_limit=None*, *sample\_rows\_limit=None*, *sample\_percentage\_limit=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/helpers/connections/connections.html#DataConnection.read)[¶](#ibm_watsonx_ai.helpers.connections.connections.DataConnection.read "Link to this definition")
Download dataset stored in remote data storage. Returns batch up to 1GB.



Parameters:
* **with\_holdout\_split** (*bool**,* *optional*) – if True, data will be split to train and holdout dataset as it was by AutoAI
* **csv\_separator** (*str**,* *optional*) – separator / delimiter for CSV file
* **excel\_sheet** (*str**,* *optional*) – excel file sheet name to use, only use when xlsx file is an input,
support for number of the sheet is deprecated
* **encoding** (*str**,* *optional*) – encoding type of the CSV
* **raw** (*bool**,* *optional*) – if False there wil be applied simple data preprocessing (the same as in the backend),
if True, data will be not preprocessed
* **binary** (*bool**,* *optional*) – indicates to retrieve data in binary mode, the result will be a python binary type variable
* **read\_to\_file** (*str**,* *optional*) – stream read data to file under path specified as value of this parameter,
use this parameter to prevent keeping data in-memory
* **number\_of\_batch\_rows** (*int**,* *optional*) – number of rows to read in each batch when reading from flight connection
* **sampling\_type** (*str**,* *optional*) – a sampling strategy how to read the data
* **sample\_size\_limit** (*int**,* *optional*) – upper limit for overall data that should be downloaded in bytes, default: 1 GB
* **sample\_rows\_limit** (*int**,* *optional*) – upper limit for overall data that should be downloaded in number of rows
* **sample\_percentage\_limit** (*float**,* *optional*) – upper limit for overall data that should be downloaded
in percent of all dataset, this parameter is ignored, when sampling\_type parameter is set
to first\_n\_records, must be a float number between 0 and 1





Note


If more than one of: sample\_size\_limit, sample\_rows\_limit, sample\_percentage\_limit are set,
then downloaded data is limited to the lowest threshold.




Returns:
one of:


* pandas.DataFrame contains dataset from remote data storage : Xy\_train
* Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame] : X\_train, X\_holdout, y\_train, y\_holdout
* Tuple[pandas.DataFrame, pandas.DataFrame] : X\_test, y\_test containing training data and holdout data from
remote storage
* bytes object, auto holdout split from backend (only train data provided)







**Examples**



```
train_data_connections = optimizer.get_data_connections()

data = train_data_connections[0].read() # all train data

# or

X_train, X_holdout, y_train, y_holdout = train_data_connections[0].read(with_holdout_split=True) # train and holdout data

```


User provided train and test data:



```
optimizer.fit(training_data_reference=[DataConnection],
              training_results_reference=DataConnection,
              test_data_reference=DataConnection)

test_data_connection = optimizer.get_test_data_connections()
X_test, y_test = test_data_connection.read() # only holdout data

# and

train_data_connections = optimizer.get_data_connections()
data = train_connections[0].read() # only train data

```





set\_client(*api\_client=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/helpers/connections/connections.html#DataConnection.set_client)[¶](#ibm_watsonx_ai.helpers.connections.connections.DataConnection.set_client "Link to this definition")
Set initialized service client in connection to enable write/read operations with connection to service.



Parameters:
**api\_client** ([*APIClient*](base.html#client.APIClient "client.APIClient")) – API client to connect to service




**Example**



```
DataConnection.set_client(api_client=api_client)

```





write(*data*, *remote\_name=None*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/helpers/connections/connections.html#DataConnection.write)[¶](#ibm_watsonx_ai.helpers.connections.connections.DataConnection.write "Link to this definition")
Upload file to a remote data storage.



Parameters:
* **data** (*str*) – local path to the dataset or pandas.DataFrame with data
* **remote\_name** (*str*) – name that dataset should be stored with in remote data storage








S3Location[¶](#s3location "Link to this heading")
-------------------------------------------------




*class* ibm\_watsonx\_ai.helpers.connections.connections.S3Location(*bucket*, *path*, *\*\*kwargs*)[[source]](_modules/ibm_watsonx_ai/helpers/connections/connections.html#S3Location)[¶](#ibm_watsonx_ai.helpers.connections.connections.S3Location "Link to this definition")
Bases: `BaseLocation`


Connection class to COS data storage in S3 format.



Parameters:
* **bucket** (*str*) – COS bucket name
* **path** (*str*) – COS data path in the bucket
* **excel\_sheet** (*str**,* *optional*) – name of excel sheet if pointed dataset is excel file used for Batched Deployment scoring
* **model\_location** (*str**,* *optional*) – path to the pipeline model in the COS
* **training\_status** (*str**,* *optional*) – path to the training status json in COS






get\_location()[[source]](_modules/ibm_watsonx_ai/helpers/connections/connections.html#S3Location.get_location)[¶](#ibm_watsonx_ai.helpers.connections.connections.S3Location.get_location "Link to this definition")




CloudAssetLocation[¶](#cloudassetlocation "Link to this heading")
-----------------------------------------------------------------




*class* ibm\_watsonx\_ai.helpers.connections.connections.CloudAssetLocation(*asset\_id*)[[source]](_modules/ibm_watsonx_ai/helpers/connections/connections.html#CloudAssetLocation)[¶](#ibm_watsonx_ai.helpers.connections.connections.CloudAssetLocation "Link to this definition")
Bases: `AssetLocation`


Connection class to data assets as input data references to batch deployment job on Cloud.



Parameters:
**asset\_id** (*str*) – asset ID of the file loaded on space on Cloud







DeploymentOutputAssetLocation[¶](#deploymentoutputassetlocation "Link to this heading")
---------------------------------------------------------------------------------------




*class* ibm\_watsonx\_ai.helpers.connections.connections.DeploymentOutputAssetLocation(*name*, *description=''*)[[source]](_modules/ibm_watsonx_ai/helpers/connections/connections.html#DeploymentOutputAssetLocation)[¶](#ibm_watsonx_ai.helpers.connections.connections.DeploymentOutputAssetLocation "Link to this definition")
Bases: `BaseLocation`


Connection class to data assets where output of batch deployment will be stored.



Parameters:
* **name** (*str*) – name of .csv file which will be saved as data asset
* **description** (*str**,* *optional*) – description of the data asset











[Next

AutoAI](autoai.html)
[Previous

Working with DataConnection](autoai_working_with_dataconnection.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [DataConnection Modules](#)
	+ [DataConnection](#dataconnection)
		- [`DataConnection`](#ibm_watsonx_ai.helpers.connections.connections.DataConnection)
			* [`DataConnection.from_studio()`](#ibm_watsonx_ai.helpers.connections.connections.DataConnection.from_studio)
			* [`DataConnection.read()`](#ibm_watsonx_ai.helpers.connections.connections.DataConnection.read)
			* [`DataConnection.set_client()`](#ibm_watsonx_ai.helpers.connections.connections.DataConnection.set_client)
			* [`DataConnection.write()`](#ibm_watsonx_ai.helpers.connections.connections.DataConnection.write)
	+ [S3Location](#s3location)
		- [`S3Location`](#ibm_watsonx_ai.helpers.connections.connections.S3Location)
			* [`S3Location.get_location()`](#ibm_watsonx_ai.helpers.connections.connections.S3Location.get_location)
	+ [CloudAssetLocation](#cloudassetlocation)
		- [`CloudAssetLocation`](#ibm_watsonx_ai.helpers.connections.connections.CloudAssetLocation)
	+ [DeploymentOutputAssetLocation](#deploymentoutputassetlocation)
		- [`DeploymentOutputAssetLocation`](#ibm_watsonx_ai.helpers.connections.connections.DeploymentOutputAssetLocation)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/autoai_working_with_dataconnection.html








Working with DataConnection - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](setup_cpd.html)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](#)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





Working with DataConnection[¶](#working-with-dataconnection "Link to this heading")
===================================================================================


Before you start an AutoAI experiment, you need to specify where your training dataset is located.
AutoAI currently supports Cloud Object Storage (COS) and data assets on Cloud.



IBM Cloud - DataConnection Initialization[¶](#ibm-cloud-dataconnection-initialization "Link to this heading")
-------------------------------------------------------------------------------------------------------------


There are three types of connections: Connection Asset, Data Asset, and Container. To upload your experiment dataset, you must initialize `DataConnection` with your COS credentials.



### Connection Asset[¶](#connection-asset "Link to this heading")



```
from ibm_watsonx_ai.autoai.helpers.connections import DataConnection, S3Location

connection_details = client.connections.create({
    client.connections.ConfigurationMetaNames.NAME: "Connection to COS",
    client.connections.ConfigurationMetaNames.DATASOURCE_TYPE: client.connections.get_datasource_type_id_by_name('bluemixcloudobjectstorage'),
    client.connections.ConfigurationMetaNames.PROPERTIES: {
        'bucket': 'bucket_name',
        'access_key': 'COS access key id',
        'secret_key': 'COS secret access key'
        'iam_url': 'COS iam url',
        'url': 'COS endpoint url'
    }
})

connection_id = client.connections.get_uid(connection_details)

# note: this DataConnection will be used as a reference where to find your training dataset
training_data_references = DataConnection(
    connection_asset_id=connection_id,
    location=S3Location(
        bucket='bucket_name',   # note: COS bucket name where training dataset is located
        path='my_path'  # note: path within bucket where your training dataset is located
        )
    )

# note: this DataConnection will be used as a reference where to save all of the AutoAI experiment results
results_connection = DataConnection(
        connection_asset_id=connection_id,
        # note: bucket name and path could be different or the same as specified in the training_data_references
        location=S3Location(bucket='bucket_name',
                            path='my_path'
                            )
    )

```




### Data Asset[¶](#data-asset "Link to this heading")



```
from ibm_watsonx_ai.autoai.helpers.connections import DataConnection

data_location = './your_dataset.csv'
asset_id = client.data_assets.get_id(asset_details = client.data_assets.create(
        name=data_location.split('/')[-1],
        file_path=data_location
        )
    )

asset_id = client.data_assets.get_id(asset_details)
training_data_references = DataConnection(data_asset_id=asset_id)

```




### Container[¶](#container "Link to this heading")



```
from ibm_watsonx_ai.autoai.helpers.connections import DataConnection, ContainerLocation

training_data_references = DataConnection(location=ContainerLocation(path="your_dataset.csv"))

```





IBM watsonx.ai software - DataConnection Initialization[¶](#ibm-watsonx-ai-software-dataconnection-initialization "Link to this heading")
-----------------------------------------------------------------------------------------------------------------------------------------


There are three types of connections: Connection Asset, Data Asset, and
FS. Note that FS is only for saving result references. To upload your experiment dataset,
you must initialize `DataConnection` with your service credentials.



### Connection Asset - DatabaseLocation[¶](#connection-asset-databaselocation "Link to this heading")



```
from ibm_watsonx_ai.autoai.helpers.connections import DataConnection, DatabaseLocation

connection_details = client.connections.create({
    client.connections.ConfigurationMetaNames.NAME: f"Connection to Database - {your_database_name} ",
    client.connections.ConfigurationMetaNames.DATASOURCE_TYPE: client.connections.get_datasource_type_id_by_name('your_database_name'),
    client.connections.ConfigurationMetaNames.PROPERTIES: {
            "database": "database_name",
            "password": "database_password",
            "port": "port_number",
            "host": "host_name",
            "username": "database_type" # e.g. "postgres"
            }
})

connection_id = client.connections.get_uid(connection_details)

training_data_references = DataConnection(
    connection_asset_id=connection_id,
    location=DatabaseLocation(
        schema_name=schema_name,
        table_name=table_name,
    )
)

```




### Connection Asset - S3Location[¶](#connection-asset-s3location "Link to this heading")


For a Connection Asset with S3Location, `connection_id` to the S3 storage is required.



```
from ibm_watsonx_ai.helpers.connections import DataConnection, S3Location

training_data_references = DataConnection(
    connection_asset_id=connection_id,
    location=S3Location(bucket='bucket_name',   # note: COS bucket name where training dataset is located
                        path='my_path'  # note: path within bucket where your training dataset is located
                        )
)

```




### Connection Asset - NFSLocation[¶](#connection-asset-nfslocation "Link to this heading")


Before establishing a connection, you need to create and start a volume where the dataset will be stored.



```
from ibm_watsonx_ai.helpers.connections import DataConnection, NFSLocation

connection_details={
        client.connections.ConfigurationMetaNames.NAME: "Client NFS Volume Connection from SDK",
        client.connections.ConfigurationMetaNames.DATASOURCE_TYPE: client.connections.get_datasource_type_id_by_name('volumes'),
        client.connections.ConfigurationMetaNames.DESCRIPTION: "NFS volume connection from python client",
        client.connections.ConfigurationMetaNames.PROPERTIES: {"instance_id": volume_id,
                                                                "pvc": existing_pvc_volume_name,
                                                                "volume": volume_name,
                                                              'inherit_access_token':"true"},
                            'flags':['personal_credentials'],
        }

client.connections.create(connection_details)

connection_id = client.connections.get_uid(connection_details)

training_data_references = DataConnection(
        connection_asset_id=connection_id,
        location = NFSLocation(path=f'/{filename}'))

```




### Data Asset[¶](#id1 "Link to this heading")



```
from ibm_watsonx_ai.helpers.connections import DataConnection

data_location = './your_dataset.csv'
asset_details = client.data_assets.create(
            name=data_location.split('/')[-1],
            file_path=data_location)

asset_id = client.data_assets.get_id(asset_details)
training_data_references = DataConnection(data_asset_id=asset_id)

```




### FSLocation[¶](#fslocation "Link to this heading")


After running `fit()`, you can store your results in a dedicated place using FSLocation.



```
from ibm_watsonx_ai.helpers.connections import DataConnection, FSLocation

# after completed run
run_details = optimizer.get_run_details()
run_id = run_details['metadata']['id']

training_result_reference = DataConnection(
    connection_asset_id=connection_id,
    location=FSLocation(path="path_to_directory")
    )

```





Batch DataConnection[¶](#batch-dataconnection "Link to this heading")
---------------------------------------------------------------------


If you use a Batch type of deployment, you can store the output of the Batch deployment using `DataConnection`.
For more information and usage instruction, see [Batch](autoai_working_with_class_and_optimizer.html#working-with-batch).



```
from ibm_watsonx_ai.helpers.connections import DataConnection, DeploymentOutputAssetLocation
from ibm_watsonx_ai.deployment import Batch

service_batch = Batch(wml_credentials, source_space_id=space_id)
service_batch.create(
        experiment_run_id="id_of_your_experiment_run",
        model="choosen_pipeline",
        deployment_name='Batch deployment')

payload_reference = DataConnection(location=training_data_references)
results_reference = DataConnection(
        location=DeploymentOutputAssetLocation(name="batch_output_file_name.csv"))

scoring_params = service_batch.run_job(
    payload=[payload_reference],
    output_data_reference=results_reference,
    background_mode=False)

```




Upload your training dataset[¶](#upload-your-training-dataset "Link to this heading")
-------------------------------------------------------------------------------------


An AutoAI experiment should be able to access your training data.
If you don’t have a training dataset stored already,
you can store it by invoking the `write()` method of the `DataConnection` object.



```
training_data_references.set_client(client)
training_data_references.write(data='local_path_to_the_dataset', remote_name='training_dataset.csv')

```




Download your training dataset[¶](#download-your-training-dataset "Link to this heading")
-----------------------------------------------------------------------------------------


To download a stored dataset, use the `read()` method of the `DataConnection` object.



```
training_data_references.set_client(client)
dataset = training_data_references.read()   # note: returns a pandas DataFrame

```








[Next

DataConnection Modules](dataconnection_modules.html)
[Previous

DataConnection](dataconnection.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [Working with DataConnection](#)
	+ [IBM Cloud - DataConnection Initialization](#ibm-cloud-dataconnection-initialization)
		- [Connection Asset](#connection-asset)
		- [Data Asset](#data-asset)
		- [Container](#container)
	+ [IBM watsonx.ai software - DataConnection Initialization](#ibm-watsonx-ai-software-dataconnection-initialization)
		- [Connection Asset - DatabaseLocation](#connection-asset-databaselocation)
		- [Connection Asset - S3Location](#connection-asset-s3location)
		- [Connection Asset - NFSLocation](#connection-asset-nfslocation)
		- [Data Asset](#id1)
		- [FSLocation](#fslocation)
	+ [Batch DataConnection](#batch-dataconnection)
	+ [Upload your training dataset](#upload-your-training-dataset)
	+ [Download your training dataset](#download-your-training-dataset)














# Text from https://ibm.github.io/watsonx-ai-python-sdk/setup_cpd.html








IBM watsonx.ai software - IBM watsonx.ai









Contents





Menu







Expand





Light mode













Dark mode






Auto light/dark mode














Hide navigation sidebar


Hide table of contents sidebar

[Skip to content](#furo-main-content)




Toggle site navigation sidebar




[IBM watsonx.ai](index.html)




Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar






[IBM watsonx.ai](index.html)





* [Installation](install.html)
* [Setup](setup.html)Toggle navigation of Setup
	+ [IBM watsonx.ai for IBM Cloud](setup_cloud.html)
	+ [IBM watsonx.ai software](#)
* [API](api.html)Toggle navigation of API
	+ [Base](base.html)
	+ [Core](core_api.html)
	+ [Federated Learning](federated_learning.html)
	+ [DataConnection](dataconnection.html)Toggle navigation of DataConnection
		- [Working with DataConnection](autoai_working_with_dataconnection.html)
		- [DataConnection Modules](dataconnection_modules.html)
	+ [AutoAI](autoai.html)Toggle navigation of AutoAI
		- [Working with AutoAI class and optimizer](autoai_working_with_class_and_optimizer.html)
		- [AutoAI experiment](autoai_experiment.html)
		- [Deployment Modules for AutoAI models](autoai_deployment_modules.html)
	+ [Foundation Models](foundation_models.html)Toggle navigation of Foundation Models
		- [Embeddings](fm_embeddings.html)
		- [Models](fm_models.html)Toggle navigation of Models
			* [Model](fm_model.html)
			* [ModelInference](fm_model_inference.html)
			* [`ModelInference` for Deployments](fm_deployments.html)
		- [Prompt Tuning](prompt_tuner.html)Toggle navigation of Prompt Tuning
			* [Working with TuneExperiment and PromptTuner](pt_working_with_class_and_prompt_tuner.html)Toggle navigation of Working with TuneExperiment and PromptTuner
				+ [Tune Experiment run](pt_tune_experiment_run.html)
				+ [Tuned Model Inference](pt_model_inference.html)
			* [Tune Experiment](tune_experiment.html)
		- [Prompt Template Manager](prompt_template_manager.html)
		- [Extensions](fm_extensions.html)
		- [Helpers](fm_helpers.html)
		- [Custom models](fm_working_with_custom_models.html)
* [Samples](samples.html)
* [Migration from `ibm_watson_machine_learning`](migration.html)
* [V1 Migration Guide](migration_v1.html)
* [Changelog](changelog.html)









[Back to top](#)



Toggle Light / Dark / Auto color theme






Toggle table of contents sidebar





IBM watsonx.ai software[¶](#ibm-watsonx-ai-software "Link to this heading")
===========================================================================



Requirements[¶](#requirements "Link to this heading")
-----------------------------------------------------


For information on how to start working with IBM watsonx.ai software, refer to [Getting started with IBM Cloud Pak for Data 4.8](https://www.ibm.com/docs/en/cloud-paks/cp-data/4.8.x?topic=getting-started-tutorials).




Supported machine learning frameworks[¶](#supported-machine-learning-frameworks "Link to this heading")
-------------------------------------------------------------------------------------------------------



For a list of supported machine learning frameworks (models) of IBM watsonx.ai software, refer to watsonx.ai documentation:* [IBM Cloud Pak for Data 4.8.x](https://www.ibm.com/docs/en/cloud-paks/cp-data/4.8.x?topic=specifications-supported-frameworks-software)
* [IBM Cloud Pak for Data 5.0.x](https://www.ibm.com/docs/en/cloud-paks/cp-data/5.0.x?topic=specifications-supported-frameworks-software)






Authentication[¶](#authentication "Link to this heading")
---------------------------------------------------------


If you are a user of IBM watsonx.ai software with IBM Cloud Pak for Data 4.8, you can create a IBM watsonx.ai Python client by providing your credentials or using a token:



Note


To determine your <APIKEY>, refer to [Generating API keys](https://www.ibm.com/docs/en/cloud-paks/cp-data/4.8.x?topic=steps-generating-api-keys).


The base URL for ibm\_watsonx\_ai comes from the cluster and add-on service instance. The URL follows this pattern:


`https://{cpd_cluster}`


`{cpd_cluster}` represents the name or IP address of your deployed cluster. For the Cloud Pak for Data system, use a hostname that resolves to an IP address in the cluster.


To find the base URL, view the details for the service instance from the Cloud Pak for Data web client. Use that URL in your requests to watsonx.ai software.



Example of creating the client using username and password credentials:



```
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials

credentials = Credentials(
                   url = "<URL>",
                   username = "<USERNAME>",
                   password = "<PASSWORD>",
                   instance_id = "openshift",
                   version = "4.8"
                  )

client = APIClient(credentials)

```


Example of creating the client using username and apikey credentials:



```
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials

credentials = Credentials(
                   url = "<URL>",
                   username = "<USERNAME>",
                   api_key = "<API_KEY>",
                   instance_id = "openshift",
                   version = "4.8"
                  )

client = APIClient(credentials)

```


Example of creating the client using a token:


**Note:** In IBM watsonx.ai software with IBM Cloud Pak for Data 4.8, you can authenticate with a token set in the notebook environment.



```
access_token = os.environ['USER_ACCESS_TOKEN']
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials

credentials = Credentials(
                   url = "<URL>",
                   token = access_token,
                   instance_id = "openshift"
                   version = "4.8"
                  )
client = APIClient(credentials)

```



**Note:*** The version value should be set to the version and release format (for example: 4.8) for the IBM watsonx.ai software you are using.
* Setting a default space ID or project ID is mandatory. For examples, refer to the `client.set.default_space()` and `client.set.default_project()` APIs in this document.



**Note for IBM watsonx.ai software with IBM Cloud Pak for Data 4.8 and above:**If the following error appears during authentication:



```
Attempt of generating `bedrock_url` automatically failed.
If iamintegration=True, please provide `bedrock_url` in wml_credentials.
If iamintegration=False, please validate your credentials.

```


There are two possible causes:


1. If you chose iamintegration=False (default configuration) during installation, the credentials might be invalid or the service is temporary unavailable.
2. If you chose iamintegration=True during installation, the autogeneration of bedrock\_url wasn’t successful. Provide the IBM Cloud Pak foundational services URL as the bedrock\_url param in the wml\_credentials.


To get the IBM Cloud Pak foundational services URL, refer to [Authenticating for programmatic access](https://www.ibm.com/docs/en/cloud-paks/cp-data/4.8.x?topic=deployments-authenticating-programmatic-access).










[Next

API](api.html)
[Previous

IBM watsonx.ai for IBM Cloud](setup_cloud.html)




 Copyright © 2023-2024, IBM
 
 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me)'s
 
 [Furo](https://github.com/pradyunsg/furo)










 On this page
 



* [IBM watsonx.ai software](#)
	+ [Requirements](#requirements)
	+ [Supported machine learning frameworks](#supported-machine-learning-frameworks)
	+ [Authentication](#authentication)















