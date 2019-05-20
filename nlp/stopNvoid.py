import spacy
import random
from spacy import displacy

TRAIN_DATA = [
        ('Check team Pls VOID Check #3018677?', [{'entities': [(27, 34, 'check_number')]},{'entities': [(15, 19, 'check_instruction')]}]), 
               ('Check Team Pls VOID Check #2983294  - Amount $24,7 ?', [{'entities': [(27, 34, 'check_number')]},{'entities': [(15, 19, 'check_instruction')]}]), 
                ('Pls VOID Check #2993435 amount $7101.00?', [{'entities': [(16, 23, 'check_number')]},{'entities': [(4, 8, 'check_instruction')]}]),
                 ('please STOP PAY on check #2962623 Check #/ EFT?', [{'entities': [(26, 33, 'check_number')]},{'entities': [(7, 11, 'check_instruction')]}]), 
                  ('Check #2990276 Stopped / Reissued', [{'entities': [(7, 14, 'check_number')]},{'entities': [(25, 32, 'check_instruction')]}]),
                   ('Check # 3027262  has been voided?', [{'entities': [(8, 15, 'check_number')]},{'entities': [(26, 30, 'check_instruction')]}]),
                    ('A - Check #2949930 Stop Paid / ReIssued?', [{'entities': [(11, 18, 'check_number')]},{'entities': [(31, 38, 'check_instruction')]}]),
                     ('Stop check #2972221?', [{'entities': [(12, 19, 'check_number')]},{'entities': [(1, 4, 'check_instruction')]}]),
                      ('Check #2885046 stopped / reissued?', [{'entities': [(7, 14, 'check_number')]},{'entities': [(25, 32, 'check_instruction')]}]), 
                       ('Stop check#2873200', [{'entities': [(11, 18, 'check_number')]},{'entities': [(1, 4, 'check_instruction')]}]), 
                        ('Stop check#2920316', [{'entities': [(11, 18, 'check_number')]},{'entities': [(1, 4, 'check_instruction')]}]), 
                         ('Check # 2870820 Stoped / Reissued', [{'entities': [(8, 15, 'check_number')]},{'entities': [(25, 32, 'check_instruction')]}])
                          ]

 


def train_spacy(data,iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
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


prdnlp = train_spacy(TRAIN_DATA, 20)

# Save our trained Model
#modelfile = input("Enter your Model Name: ")
prdnlp.to_disk('modelfile')

#Test your text
#test_text = input("Enter your testing text: ")
doc = prdnlp('#pls. stop the check# 2870820 Stoped / Reissued')
#Check # 2870820 Stoped / Reissued
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    
displacy.serve(doc,style="ent")