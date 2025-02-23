from bert_toxic_chat_clf.config.configuration import ConfigurationManager
from bert_toxic_chat_clf.components.trainer import Trainer
from bert_toxic_chat_clf import logger

STAGE_NAME = 'training'

class TrainingPipeline:

    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        trainer = Trainer(config=training_config)
        trainer.setup_device()
        trainer.setup_model()
        trainer.setup_optimizer()
        trainer.split_data()
        trainer.train()
        trainer.run_validation()


if __name__ == '__main__':
    
    try:
        logger.info(f'Stage {STAGE_NAME} started')
        obj = TrainingPipeline()
        obj.main()
        logger.info(f'{STAGE_NAME} completed successfully...')
    
    except Exception as e:
        logger.exception(e)
        raise e