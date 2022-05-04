# Responsible AI - with great powers comes great responsibility

## What is Responsible AI

Responsible AI is a AI Governance framework.  With increased use of AI / ML in critical decision making process, Responsible AI provides a objective index for measuring the health of the model across the following dimensions

- Security:  Who can access the Model in production
- Differential Privacy:  Differential Privacy of data used in model training exercise
- Interpretability:  Ability of the model to be interpretable
- Consistency:  Model consistency across multiple test data sets
- Transparency:  Interpretability of the Model
- Bias and Fairness: Bias and Fairness of the Model
- Environment friendly:  Carbon Emissions of the Model training / Inference

## How does it work

1. Build the Model using framework of your choice.  
2. Import the Responsible AI package (Python package)
3. Add the details of the Responsible AI components of your model to Responsible AI Framework
4. Responsible AI calculates an Index (float) that denotes the health of the model.  Higher the number better the model

## Support

At the time of this writing, the MVP of the Responsible AI framework supports the following dimensions

- Differential Privacy
- Interpretability
- Bias (Imbalance class)
- Environment Friendly (Carbon emissions)

You can add a bunch of Models for a specific use case to the Responsible AI framework and Responsible AI ranks them on the basis of its over health or health per dimension

![image](https://user-images.githubusercontent.com/7538839/166708253-8431e3a9-a277-445c-8536-ea341a45553d.png)

You can add a bunch of Models for a specific use case to the Responsible AI framework and Responsible AI ranks them on the basis of its over health or health per dimension

## Pre-requisistes

1. Emission tracking is calcualted using [Code Carbon](https://codecarbon.io/).  Install Code Carbon using

>     !pip install codecarbon

2.Differential privacy is calculated usig [Opacus](https://opacus.ai/).  Install Opacus

>     !pip install opacus

2.Interpretability is calculated usig [Captum](https://captum.ai/).  Install Captum

>     !pip install captum

3.Responsible index is calcualted using ResponsibleML.  Install ResponsibleML

>     !pip install -i https://test.pypi.org/simple/ responsibleML

## How it works

1. AI Governance is the overarching framework for responsible AI
2. Responsible AI is across 7 dimensions - Security, Privacy, Explainability, Trust, bias and Emissions
3. Current Framework support the above 4 for calculating RAI index
4. Responsible AI has
    a.  Model List (list of all models that you created for calculating RAI Index)
    b.  Models - every model created by you for the use case.

## Code sample

1. Import the package in your code

>     from aigovernance import responsibleML

2.Initiate a Model and a Model List

>     r_model = responsibleML.responsibleModel(modelname, modeltype)
>     model_list = responsibleML.models()

3.Bias Information - During data engineering phase of your ML life cycle, send the label data frame (Y) to responsibleModel

>     r_model.calcualte_bias(label_df)

4.Carbon Emissions - For calculating carbon emissions, before you start your model training, start the tracker.  Once the model training is complete, stop the tracker

>     r_model.track_emissions()
> 
>     << your model training>>
> 
>     r_model.stop_tracking()

5.Model Interpretability - Ability of the model to be interpretable by the top 3 features by > 70%

>     r_model.intrepret(input_tensor, model,target_class)

6.Differential Privacy -

>     priv_model, priv_opt, priv_datasetloader = r_model.privatize(dp_model, optimizer, train_dataloader, noise_multiplier, max_per_sample_grad_norm) 
>     r_model.calculate_privacy_score(delta=1e-5)

7.Responsible Index:  You can now retrieve the responsible index of the model using

>     responsible_index = r_model.rai_index()

8.Given the nature of ML, you will be building multiple models.  Use steps 2 to 7 initiate the model with responsible dimension.  You can now add the models you created on to a model list

>     model_list.add_model(r_model)

9.You can list all the models and their responsible score using

>     model_list.list_models()

10.finally, to identify the most responsible model that you can use for your production use case, you can get the models ranked by their responsibility index
>
>     model_list.rank_models()
