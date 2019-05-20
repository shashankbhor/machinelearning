# -*- coding: utf-8 -*-

class Training:
    def __init__(self):
        print('inside')
        
    def buildModel(self,modelConfiguration):
        print(modelConfiguration)
        if not isinstance(modelConfiguration,ModelConfiguration):
            raise Exception('The model configuration is not set')
            return
        else:
            print('Starting Model configuration..')
        
        TRAIN_DATA = data
        nlp = spacy.blank('en')  # create blank Language class
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner, last=True)
       
        for  _,tagged_set in TRAIN_DATA:
            for tagged_item in tagged_set:
                for item in tagged_item.get('entities'):
                    ner.add_label(item[2])
                

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.begin_training()
            for itn in range(20):
                random.shuffle(TRAIN_DATA)
                losses = {}
                for text, annotations in TRAIN_DATA:
                    print(annotations)
                    nlp.update(
                        [text,text],  # batch of texts
                        annotations,  # batch of annotations
                        drop=0.2,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
                print(losses) 
        return nlp

        

        
        