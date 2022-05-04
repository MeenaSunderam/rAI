import json
import numpy as np
import logging
from codecarbon import EmissionsTracker as ET
from opacus import PrivacyEngine 
from captum.attr import IntegratedGradients

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class responsibleModel:
    
    __modelname__ = ""
    __framework__ = ""
    __emissions__ = 0.0
    __classbalance__ = 0.0
    __interpretable_degree__ = 0.0
    __epsilon__ = 0.0
    __tracker__ = None
    __privacy_engine__ = None
    
    def __init__(self,):
        self.__modelname__ = ""
        self.__framework__ = ""
        self.__emissions__ = 0.0
        self.__classbalance__ = 0.0
        self.__interpretable_degree__ = 0.0
        self.__epsilon__ = 0.0
        
        self.__tracker__ = ET(project_name = "",
            measure_power_secs = 15,
            save_to_file = False)
        
        self.__privacy_engine__ = PrivacyEngine()

    def __init__(self, 
                 modelname: str,
                 framework:str,
                 interpretable_degree:float = 0.0,
                 emissions:float = 0.0,
                 classbalance:float= 0.0,
                 epsilon:float = 0.0):
        
        self.__modelname__ = modelname
        self.__framework__ = framework
        self.__emissions__ = emissions
        self.__classbalance__ = classbalance
        self.__interpretable_degree__ = interpretable_degree
        self.__epsilon__ = epsilon
    
        self.__tracker__ = ET(project_name = modelname,
            measure_power_secs = 15,
            save_to_file = False)
        
        self.__privacy_engine__ = PrivacyEngine()
    
    def set_interpretability(self, interpretable_degree: float):
        self.__interpretable_degree__ = interpretable_degree

    def set_emissions(self, carbon_emissions: float):
        self.__emissions__ = carbon_emissions

    def set_classbalance(self, minclass: float):
        self.__classbalance__ = minclass

    def set_epsilon(self, privacy_epsilon: bool):
        self.__epsilon__ = privacy_epsilon

    def set_framework(self, framework: str):
        self.__framework__ = framework
        
    def calculate_emissions_index(self):

        if self.__emissions__ <= 500:
            emissionIndex = 3
        elif self.__emissions__ > 500 and self.__emissions__ <= 10000:
            emissionIndex = 2
        else:
            emissionIndex = 1

        return emissionIndex

    def calculate_privacy_index(self):
        if self.__epsilon__ <= 1:
            privacyIndex = 3
        elif self.__epsilon__ > 1 and self.__epsilon__ <= 10:
            privacyIndex = 2
        else:
            privacyIndex = 1

        return privacyIndex

    def calculate_interpretability_index(self):

        interIndex = 1

        if self.__interpretable_degree__ > .70:
            interIndex = 3
        elif self.__interpretable_degree__ > .50 and self.__interpretable_degree__ < .70:
            interIndex = 2
        else:
            interIndex = 1

        return interIndex

    def calculate_bias_index(self):
        
        if self.__classbalance__ >= 0.4:
            bindex = 3
        elif self.__classbalance__ > 0.2 and self.__classbalance__ < 0.4:
            bindex = 2
        else:
            bindex = 1

        return bindex
    
    def describe_model(self):
        value = json.dumps({"model name": self.__modelname__,
                    "framework": self.__framework__,
                    "emissions": self.__emissions__,
                    "interpretability": self.__interpretable_degree__,
                    "privacy": self.__epsilon__,
                    "bias": self.__classbalance__,})        
        return value
    
    def model_rai_components(self):
        
        emission_index = self.calculate_emissions_index()
        privacy_index = self.calculate_privacy_index()
        bias_index = self.calculate_bias_index()
        interpret_index = self.calculate_interpretability_index()
        RAI_index = self.rai_index()
        
        value = json.dumps({"model name": self.__modelname__,
                            "framework": self.__framework__,
                            "rai index": RAI_index,
                            "emission_index": emission_index,
                            "privacy_index": privacy_index,
                            "bias_index": bias_index,
                            "interpretability_index": interpret_index})

        return value
        
    def rai_index(self):
    
        index = 0.0
        weights = 0.25

        emission_index = self.calculate_emissions_index()
        privacy_index = self.calculate_privacy_index()
        bias_index = self.calculate_bias_index()
        interpret_index = self.calculate_interpretability_index()

        index = weights * (emission_index + privacy_index + bias_index + interpret_index)

        return index

    def track_emissions(self):
        # Calculate Emissions
        self.__tracker__.start()
        
    def stop_tracking(self):
        self.__emissions__ =  self.__tracker__.stop()
        
    def calculate_bias(self, df_label: str):
        
        # Get the number of classes & samples
        label_classes = df_label.value_counts(ascending=True)
        totalvalues = label_classes.sum()
        min_class_count = label_classes.values[1]
        
        #calcualte the bias
        self.__classbalance__ = min_class_count / totalvalues
        
    def privatize(self, model, optimizer, dataloader, noise_multiplier, max_grad_norm):
        
        model, optimizer, dataloader = self.__privacy_engine__.make_private(module=model,
                                                                            optimizer=optimizer,
                                                                            data_loader=dataloader,
                                                                            noise_multiplier = noise_multiplier,
                                                                            max_grad_norm= max_grad_norm)

        return model, optimizer, dataloader
        
    def calculate_privacy_score(self, delta):
        self.__epsilon__ = self.__privacy_engine__.get_epsilon(delta)
    
    def interpret(self, input_tensor, model,target_class):
        
        ig = IntegratedGradients(model)
        input_tensor.requires_grad_()
        attr, delta = ig.attribute(input_tensor,target=target_class, return_convergence_delta=True)
        attr = attr.detach().numpy()
        importance = np.mean(attr, axis=0)
        
        importance = np.abs(importance)        
        importance[::-1].sort()
        
        total_weightage = np.sum(importance)
        key_features_weightage = importance[0] + importance[1] + importance[2]
        
        __interpretable_degree__ = key_features_weightage / total_weightage
            
class models:
    model_list = []
    
    def __init__(self):
        self.model_list = []
    
    def add_model(self, modelname, framework, intrepretability, emissions, bias, epsilon):
        model = responsibleModel(modelname, framework, intrepretability, emissions, bias, epsilon)
        self.model_list.append(model)
        
    def add_model(self, model):
        self.model_list.append(model)
        
    def remove_model(self, modelname):
        self.model_list.remove(modelname)
        
    def list_models(self):
        model_json = ""
        for model in self.model_list:
            model_json += model.describe() 
            if model != self.model_list[-1]:
                model_json += ","
                                
            model_json += "\n"
            
        model_json = "[" + model_json + "]"
        
        return model_json
    
    def get_model(self, modelname):
        for model in self.model_list:
            if model.__modelname__ == modelname:
                return model
        return None
    
    def rank_models(self, rank_by = "rai_index"):
        sorted_json = ""
        
        if rank_by == "rai_index":
            sorted_models = sorted(self.model_list, key=lambda x: x.rai_index(), reverse=True)
        elif rank_by == "emissions":
            sorted_models = sorted(self.model_list, key=lambda x: x.calculate_emissions_index(), reverse=True)
        elif rank_by == "privacy":
            sorted_models = sorted(self.model_list, key=lambda x: x.calculate_privay_index(), reverse=True)
        elif rank_by == "bias":
            sorted_models = sorted(self.model_list, key=lambda x: x.calculate_bias_index(), reverse=True)
        elif rank_by == "interpretability":
            sorted_models = sorted(self.model_list, key=lambda x: x.calculate_interpretability_index(), reverse=True)
            
        for model in sorted_models:
            sorted_json += model.model_rai_components()
            if(model != sorted_models[-1]):
                sorted_json += ","
            sorted_json += "\n"
            
        sorted_json = "[" + sorted_json + "]"
        return sorted_json