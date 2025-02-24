from bert_toxic_chat_clf.config.configuration import ConfigurationManager
from bert_toxic_chat_clf.components.evaluation import Evaluation
from bert_toxic_chat_clf import logger

STAGE_NAME = 'evaluation'

class EvaluationPipeline():

    def __init__(self):
        pass


    def main():
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(evaluation_config)
        evaluation.load_model()
        evaluation.load_data()
        evaluation.run_validation()
        evaluation.log_mlflow()

if __name__ == '__main__':
    
    try:
        logger.info(f'Stage {STAGE_NAME} started')
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f'{STAGE_NAME} completed successfully...')
    
    except Exception as e:
        logger.exception(e)
        raise e