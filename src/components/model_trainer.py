import os
import sys
from dataclasses import dataclass
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from tensorflow.keras.regularizers import l2
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer__config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:

            logging.info(f"Received train_array type: {type(train_array)}, test_array type: {type(test_array)}")

            # Ensure arrays are dense for slicing
            if hasattr(train_array, "toarray"):
                train_array = train_array.toarray()
            if hasattr(test_array, "toarray"):
                test_array = test_array.toarray()

            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )  

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            } 

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            
                },
                "Random Forest Regressor":{
                
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "Linear Regression":{},
                "Lasso":{},
                "Ridge":{},
                "K-Neighbors Regressor":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models, param=params)

            ## To get best model score from dict
            best_model_score = max(model_report.values())

            ## To get best model name from dict

            best_model_name = max(model_report, key=model_report.get)

            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with score {best_model_score}")

            if best_model_score<0.6:
                raise CustomException("No suitable model found",sys)
        
        
            logging.info("Saving the best model")

            save_object(
                file_path=self.model_trainer__config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            logging.error(f"Error in ModelTrainer: {e}")
            raise CustomException(e, sys)
