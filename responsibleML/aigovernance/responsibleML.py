import numpy as np
import pandas as pd
import json

import shap
from codecarbon import EmissionsTracker
from opacus import PrivacyEngine 
from captum.attr import IntegratedGradients

class responsible_model:
    __model_name = None
    __framework = 'sklearn'
    
    __emissions = None
    __class_balance  = None
    __epsilon = None
    __interpretability = None
    
    __emissions_index = 0
    __bias_index = 0
    __privacy_index = 0
    __interpretability_index = 0
    
    index_weightage = "EQUAL"
    
    ### EmissionsTracker ###
    __tracker = None
    
    def __init__(self, model_name):
        
        # General Model inforamtion
        self.__model_name = model_name
        self.__framework = "sklearn"
        
        # Responsible Model Metrics
        self.__emissions = None
        self.__class_balance = None
        self.__epsilon = None
        self.__interpretability = None
        
        # Responsible Index
        self.__emissions_index = None
        self.__bias_index = None
        self.__privacy_index = None
        self.__interpretability_index = None
        
        # Overall Responsible Index
        self.__model_index = None
        
        #self.__tracker = ET()
        
    def get_model_name(self):
        return self.__model_name
    
    def get_framework(self):
        return self.__framework
    
    def get_emissions(self):
        return self.__emissions
    
    def get_class_balance(self):
        return self.__class_balance
    
    def get_epsilon(self):
        return self.__epsilon
    
    def get_interpretability(self):
        return self.__interpretability

    def get_emissions_index(self):
        if self.__emissions_index is None:
            self.__calculate_emissions_index()
            
        return self.__emissions_index
    
    def get_interpretability_index(self):
        if self.__interpretability_index is None:
            self.__calculate_interpretability_index()
        
        return self.__interpretability_index
    
    def get_bias_index(self):
        if self.__bias_index is None:
            self.__calculate_bias_index()
            
        return self.__bias_index
    
    def get_privacy_index(self):
        if self.__privacy_index is None:
            self.__calculate_privacy_index()
            
        return self.__privacy_index
    
    def set_model_name(self, model_name):
        self.__model_name = model_name
        
    def set_framework(self, framework):
        self.__framework = framework
        
    def set_emissions(self, emissions):
        self.__emissions = emissions
        
    def set_class_balance(self, class_balance):
        self.__class_balance = class_balance
        
    def set_epsilon(self, epsilon):
        self.__epsilon = epsilon
    
    def set_interpretability(self, interpretability):
        self.__interpretability = interpretability
        
    def get_model_info(self):
        
        value = json.dumps({"model name": self.__model_name,
                    "framework": self.__framework,
                    "emissions": self.__emissions,
                    "class_balance": self.__class_balance,
                    "interpretability": self.__interpretability,
                    "epsilon": self.__epsilon,
                    "bias Index": self.__bias_index,
                    "privacy index": self.__privacy_index,
                    "interpretability index": self.__interpretability_index,
                    "emission index": self.__emissions_index,
                    "model_rai_index": self.__model_index})
        
        return value
                    
    def get_model_info_json(self):
        return json.dumps(self.get_model_info())
    
    ### ---------- Emissions Index ---------- ###
    
    def start_emissions_tracker(self):
        self.__tracker = EmissionsTracker()
        self.__tracker.start()
    
    def stop_emissions_tracker(self):
        self.__emissions : float = self.__tracker.stop()
        
    def __calculate_emissions_index(self):
        if self.__emissions <= 500:
            self.__emissions_index = 3
        elif self.__emissions > 500 and self.emissions <= 10000:
            self.__emissions_index = 2
        else:
            self.__emissions_index = 1
        
    ### ---------- Bias Index ---------- ###
    
    def calculate_bias(self, df_label: pd.DataFrame):
        # Get the number of classes & samples
        label_classes = df_label.value_counts(ascending=True)
        
        totalvalues = label_classes.sum()
        min_class_count = label_classes.values[0]
        
        #calcualte the bias
        self.__class_balance = min_class_count / totalvalues
    
    def __calculate_bias_index(self):
        if self.__class_balance >= 0.4:
            self.__bias_index = 3
        elif self.__class_balance > 0.2 and self.__class_balance < 0.4:
            self.__bias_index = 2
        else:
            self.__bias_index = 1
    
    ### ---------- Privacy Index ---------- ###    
    
    def calculate_privacy(self):
        return
    
    def __calculate_privacy_index(self):
        if self.__epsilon <= 1:
            self.__privacy_index = 3
        elif self.__epsilon > 1 and self.__epsilon < 10:
            self.__privacy_index = 2
        else:
            self.__privacy_index = 1
    
    ### ---------- Interpretability Index ---------- ###        
    
    def calculate_interpretability(self, model_type, model, df_x):

        # Explain model predictions using shap library:
        shape_values_df = None
        
        if model_type == 'linear':
            explainer = shap.LinearExplainer(model, df_x, feature_dependence="interventional")
            shap_values = explainer.shap_values(df_x)
            shape_values_df = pd.DataFrame(shap_values, columns=df_x.columns)     
            
        elif model_type == 'treebased':        
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_x)
            shape_values_df = pd.DataFrame(shap_values[1], columns=df_x.columns)
            
        vals = np.abs(shape_values_df.values).mean(0)   
        sorted_vals = np.sort(vals, axis=0)        
        top3 = sorted_vals[-3:].sum()
        total = sorted_vals.sum()
        
        self.__interpretability = top3 / total
    
    def __calculate_interpretability_index(self):
        
        if self.__interpretability >= 0.6:
            self.__interpretability_index = 3
        elif self.__interpretability > 0.4 and self.__interpretability < 0.6:
            self.__interpretability_index = 2
        else:
            self.__interpretability_index = 1
    
    ### ---------- Responsible Model Index ---------- ###                
    
    def get_model_index(self):
        self.__calculate_emissions_index()
        self.__calculate_bias_index()
        self.__calculate_privacy_index()
        self.__calculate_interpretability_index()
        
        if self.index_weightage == "EQUAL":
            self.__model_index = (self.__emissions_index + self.__bias_index + self.__privacy_index + self.__interpretability_index) / 4
        
        return self.__model_index
    
class pytorch_model(responsible_model):
    
    def __init__(self, model_name):
        super().__init__(model_name)
        super().set_framework('pytorch')
        
    ### ---------- Privacy Index ---------- ###    
    
    def privatize(self, model, optimizer, dataloader, noise_multiplier, max_grad_norm):
        
        model, optimizer, dataloader = self.__privacy_engine__.make_private(module=model,
                                                                            optimizer=optimizer,
                                                                            data_loader=dataloader,
                                                                            noise_multiplier = noise_multiplier,
                                                                            max_grad_norm= max_grad_norm)

        return model, optimizer, dataloader
        
    def calculate_privacy(self, delta):
        self.__epsilon__ = self.__privacy_engine__.get_epsilon(delta)
    
    ### ---------- Overwrite Interpretability Index ---------- ###        
    def calculate_interpretability(self, input_tensor, model,target_class):

        ig = IntegratedGradients(model)
        input_tensor.requires_grad_()
        attr, delta = ig.attribute(input_tensor,target=target_class, return_convergence_delta=True)
        attr = attr.detach().numpy()
        importance = np.mean(attr, axis=0)
        
        importance = np.abs(importance)        
        importance[::-1].sort()
        
        total_weightage = np.sum(importance)
        key_features_weightage = importance[0] + importance[1] + importance[2]
        
        super().set_interpretability = key_features_weightage / total_weightage
        
class rai_models:
    model_list = []
    
    def __init__(self):
        self.model_list = []
        
    def add_model(self, model):
        self.model_list.append(model)
        
    def remove_model(self, modelname):
        self.model_list.remove(modelname)
        
    def list_models(self):
        model_json = ""
        for model in self.model_list:
            model_json += model.get_model_info() 
            if model != self.model_list[-1]:
                model_json += ","
                                
            model_json += "\n"
            
        model_json = "[" + model_json + "]"
        
        return model_json
    
    def get_model(self, modelname):
        for model in self.model_list:
            if model.get_model_name() == modelname:
                return model
        return None
    
    def rank_models(self, rank_by = "rai_index"):
        sorted_json = ""
        
        if rank_by == "rai_index":
            sorted_models = sorted(self.model_list, key=lambda x: x.get_model_index(), reverse=True)
        elif rank_by == "emissions":
            sorted_models = sorted(self.model_list, key=lambda x: x.get_emissions_index(), reverse=True)
        elif rank_by == "privacy":
            sorted_models = sorted(self.model_list, key=lambda x: x.get_privacy_index(), reverse=True)
        elif rank_by == "bias":
            sorted_models = sorted(self.model_list, key=lambda x: x.get_bias_index(), reverse=True)
        elif rank_by == "interpretability":
            sorted_models = sorted(self.model_list, key=lambda x: x.get_interpretability_index(), reverse=True)
            
        for model in sorted_models:
            sorted_json += model.model_rai_components()
            if(model != sorted_models[-1]):
                sorted_json += ","
            sorted_json += "\n"
            
        sorted_json = "[" + sorted_json + "]"
        return sorted_json
