import spacy
import random
 
 
TRAIN_DATA = [('Check team Pls VOID Check #3018677?', {'entities': [(27, 34, 'check_number')]}),
               ('Check Team Pls VOID Check #2983294  - Amount $24,7 ?', {'entities': [(27, 34, 'check_number')]}),
                ('Pls VOID Check #2993435 amount $7101.00?', {'entities': [(16, 23, 'check_number')]}),
                 ('please STOP PAY on check #2962623 Check #/ EFT?', {'entities': [(26, 33, 'check_number')]}),
                  ('Check #2990276 Stopped / Reissued', {'entities': [(7, 14, 'check_number')]}),
                   ('Check # 3027262  has been voided?', {'entities': [(8, 15, 'check_number')]}),
                    ('A - Check #2949930 Stop Paid / ReIssued?', {'entities': [(11, 18, 'check_number')]}),
                     ('Stop check #2972221?', {'entities': [(12, 19, 'check_number')]}),
                      ('Check #2885046 stopped / reissued?', {'entities': [(7, 14, 'check_number')]}),
                       ('Stop check#2873200', {'entities': [(11, 18, 'check_number')]}),
                        ('Stop check#2920316', {'entities': [(11, 18, 'check_number')]}),
                         ('Check # 2870820 Stoped / Reissued', {'entities': [(8, 15, 'check_number')]}
                          )]
 
 
def train_spacy(data,iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
      
 
    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])
 
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(20):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                print(annotations)
                '''
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)'''
    return nlp
 
 
prdnlp = train_spacy(TRAIN_DATA, 20)
 
# Save our trained Model
#modelfile = input("Enter your Model Name: ")
prdnlp.to_disk('modelfile')
 
#Test your text
#test_text = input("Enter your testing text: ")
doc = prdnlp('PLEASE VOID CHECK PAYMENT TO INSURED AND REISSUE T ')
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
 