# -*- coding: utf-8 -*-

import spacy
import random
from spacy import displacy
import json

TRAIN_DATA = [
('Check team Pls VOID Check #3018677?',{'entities': [(27, 34, 'check_number'),(15, 19, 'check_instruction')]}),
('Please stop pay check Vehicle is a total loss Check #/ EFT #: 000002861786 ',{'entities': [(62, 74, 'check_number'),(7, 11, 'check_instruction')]}),
('Pls stop pymt and reissue.; Pls overnight, no ignature required.; Thx; Check #/ EFT #: 000002989352',{'entities': [(87, 99, 'check_number'),(18, 25, 'check_instruction')]}),
('Please verify if check has cleared, if not please stop pay on check, I will reissue overnight mail , please send me task to issue check overnight mail after stop confirmed   ;  thank you;Check #/ EFT #: 000002968167',{'entities': [(203, 215, 'check_number'),(76, 83, 'check_instruction')]}),
('CA Assist- pls pull and void check;#/ EFT #: 000003006683',{'entities': [(45, 57, 'check_number'),(15, 19, 'check_instruction')]}),
('Please stop payment to claimant. Reissue to the address listed below.;Check #/ EFT #: 000002965680',{'entities': [(86, 98, 'check_number'),(33, 40, 'check_instruction')]}),
('please place stop pay on check;Please notify me once complete as I will need to re-issue, thank you;Check #/ EFT #: 000002968603',{'entities': [(116, 128, 'check_number'),(80, 88, 'check_instruction')]}),
('Please stop pay check 2983409 and reissue to:',{'entities': [(22, 29, 'check_number'),(34, 41, 'check_instruction')]}),
('stop pay and reissue to same address on check',{'entities': [(62, 74, 'check_number'),(13, 20, 'check_instruction')]}),
('Please stop pay and re-issue check to insd Check #/ EFT #: 000002972132',{'entities': [(59, 61, 'check_number'),(20, 28, 'check_instruction')]}),
('Please void and reissue Check #/ EFT #: 000003027463 ',{'entities': [(40, 52, 'check_number'),(16, 23, 'check_instruction')]}),
('Sent request to check team to pull &amp; void check Check #/ EFT #: 000002947196 ',{'entities': [(68, 80, 'check_number'),(41, 45, 'check_instruction')]}),
('Please VOID this payment!....Advise me;when it is done....Thanks! Check #/ EFT #: 000002959286',{'entities': [(203, 215, 'check_number'),(76, 83, 'check_instruction')]}),
('pls stop pay and reissue the check to the insured',{'entities': [(17, 24, 'check_instruction'),(4, 8, 'check_instruction')]}),
('requested check be pulled for void Check #/ EFT #: 000002918001',{'entities': [(51, 63, 'check_number'),(30, 34, 'check_instruction')]}),
('Please stop pay check #2863602 and reissue. Shop claims they have not recieved it',{'entities': [(23, 30, 'check_number'),(35, 42, 'check_instruction')]}),
('please stop payment on the following check and reissue',{'entities': [(7, 11, 'check_instruction'),(47, 54, 'check_instruction')]}),
('Check void for Incorrect payee.',{'entities': [(6, 10, 'check_instruction')]}),
('Please void payment and reissue to Micheala Barton Check #/ EFT #: 000002962873',{'entities': [(67, 79, 'check_number'),(24, 31, 'check_instruction')]}),
('Check Team, please pull check below, need to void',{'entities': [(19, 23, 'check_instruction'),(45, 48, 'check_instruction')]}),
('Has this check been cashed? If not, please stop pay. Thank you Check #/ EFT #: 000002871096',{'entities': [(79, 91, 'check_number'),(43, 47, 'check_instruction')]}),
('Pls stop pymt and reissue.; Pls overnight, no signature required',{'entities': [(4, 8, 'check_instruction'),(18, 25, 'check_instruction')]}),
('Please stop payment to claimsnet and reissue to claimant directly.Check #/ EFT #: 000002950130',{'entities': [(82, 94, 'check_number'),(37, 44, 'check_instruction')]}),
('please stop pay this check 000002847430',{'entities': [(27, 39, 'check_number'),(7, 11, 'check_instruction')]}),
('pls stop-pay and reissue the below checks, Check #/ EFT #: 000002932568 and #8765437',{'entities': [(58, 70, 'check_number'),(76, 83, 'check_number'),(17, 24, 'check_instruction')]}),
('Please stop pay check #3006941 and re-issue ',{'entities': [(23, 30, 'check_number'),(35, 43, 'check_instruction')]}),
('issued check with incorrect amount $13,654.84  please void check; Check #/ EFT #: 000002872436',{'entities': [(84, 96, 'check_number'),(54, 58, 'check_instruction')]}),
('Please pull check below/void check below and reissue check to claimant - Noelle Levi direct and mail to claimant at 1944 Mark St, Santa Ana, California 92703;for same amount; Check #/ EFT #:   000002935387', {'entities': [(155, 167, 'check_number'),(45, 52, 'check_instruction')]}),
('pls stop pymt',{'entities': [(4, 8, 'check_instruction')]}),
('Check need to be Stop and Reissue',{'entities': [(18, 22, 'check_instruction'),(27, 34, 'check_instruction')]}),
('Check number is 000008989787 please void the it and reissue with 1200USD',{'entities': [(16, 28, 'check_number'),(52, 59, 'check_instruction')]}),
('please ask check team to void the Check number 000008989787 and advise to sent a ack. so that i can reissue',{'entities': [(47, 59, 'check_number'),(100, 107, 'check_instruction')]}),
('check number is #8989890 which is needed to stop',{'entities': [(17, 24, 'check_number'),(44, 48, 'check_instruction')]}),
('The amount was printed wrong,so plwase stop the check 6824720 ',{'entities': [(54, 61, 'check_number'),(39, 43, 'check_instruction')]}),
('the check #000002345432 mailed to wrong address please void it',{'entities': [(11, 23, 'check_number'),(55, 59, 'check_instruction')]}),
('below check needs to be stop and reissue #3456789 and # 1245780',{'entities': [(42, 49, 'check_number'),(56, 63, 'check_number'),(33, 40, 'check_instruction')]}),
('Please stop the payment of check #2323239 and # 0000001234567 and reissue with new amount',{'entities': [(48, 60, 'check_number'),(34, 41, 'check_number'),(66, 73, 'check_instruction')]}),
('pull back the check #000006575890 and reissue',{'entities': [(21, 33, 'check_number'),(38, 45, 'check_instruction')]}),
('Check #4545673 need to stop',{'entities': [(7, 14, 'check_number'),(23, 27, 'check_instruction')]}),
('stop and re-issue the following checks #000004545673 and #000009876543 ',{'entities': [(40, 52, 'check_number'),(58, 60, 'check_number'),(9, 17, 'check_instruction')]}),
('Please call the mail services and stop the check',{'entities': [(34, 38, 'check_instruction')]}),
('Call the check team and ask for the status of the check #000003434542',{'entities': [(57, 69, 'check_number')]}),
('Check #000002345678 is missing so please void it',{'entities': [(7, 19, 'check_number'),(41, 44, 'check_instruction')]}),
('please help! check #8989890 was asked to void and re-issue',{'entities': [(20, 32, 'check_number'),(50, 58, 'check_instruction')]}),
('please call for the check detailes',{'entities': []}),
('I want information for below check ',{'entities': []}),
('please talk to check team for the details',{'entities': []}),
('the mailing address for the check is not correct',{'entities': []}),
('the check was returned',{'entities': []}),
('Please reissue the check #00000547892',{'entities': [(26, 38, 'check_number'),(7, 14, 'check_instruction')]}),
('Please pull and VOID Check ',{'entities': [(16, 20, 'check_instruction')]}),
(' iv now totaled, please stop pay the original ck for repairs ',{'entities': [(24, 28, 'check_instruction')]}),
('ACS, pls stop pay and reissue the check to the insured. Thanks',{'entities': [(22, 29, 'check_instruction')]}),
('Support,   ;  Please stop pay and reissue check 2986939 to corrected clmt business name ',{'entities': [(48, 55, 'check_number'),(34, 41, 'check_instruction')]}),
('Natalie, Per our discussion please void following chk   ',{'entities': [(35, 39, 'check_instruction')]}),
(' tls,   Pls stoppay on below check and reissue.   ;',{'entities': [(39, 46, 'check_instruction')]}),
('CA Support,  ;  Please void and reissue payment overnight made payable to:   Eastern North Carolina Broadcasting Corporation          Check #/ EFT #:   000002915176',{'entities': [(152, 164, 'check_number'),(32, 39, 'check_instruction')]}),
(' please void below payment issued to;insured error for check #000001234231',{'entities': [(62, 74, 'check_number'),(8, 15, 'check_instruction')]}),
('Please stop pay and reissue (Overnight Request)          Check #/ EFT #:   000002944235',{'entities': [(76, 88, 'check_number'),(21, 28, 'check_instruction')]}),
('Please pull to void',{'entities': [(15, 19, 'check_instruction')]}),
('No record found for Check #2992304  Your action has been successfully submitted  and will be reissue  ',{'entities': [(27, 34, 'check_number'),(93, 100, 'check_instruction')]}),
('ACS, void &amp; reissue cv payment please  the payment needs to have Apartment F in the address Check #/ EFT #: 000002869923 ',
{'entities': [(112, 124, 'check_number'),(16, 23, 'check_instruction')]}),
('Per note from ATS. MAPS issued supplement for incorrect amount. Please place stop payment on the following check:#/ EFT #: 000002983261',{'entities': [(123, 135, 'check_number'),(77, 84, 'check_instruction')]}),
('PLEASE VOID;THE CHECK THAT; WAS ISSUED FOR CV',{'entities': [(7, 11, 'check_instruction')]}),
('PLEASE STOP PAY AND RE-ISSUE Check #/ EFT #: 000002953562',{'entities': [(45, 57, 'check_number'),(20, 28, 'check_instruction')]}),
('PLEASE STOP PAY AND REISSUE THE CHECKS AS FOLLOWS: ',{'entities': [(20, 27, 'check_instruction')]}),
('reissue the check #000001231231 ',{'entities': [(19, 31, 'check_number'),(0, 7, 'check_instruction')]}),
('Support, Please stop pay check 3006566 and reissue as payable to',{'entities': [(31, 38, 'check_number'),(43, 50, 'check_instruction')]}),
('Please do a stop pay on the two checks sent out for Claimant repairs  ',{'entities': [(12, 19, 'check_instruction')]}),
('CA - please stop and re-issue to clmnt check to new address provided below  ;  Thank you',{'entities': [(21, 29, 'check_instruction')]}),
('requests stop pay and reissue Check #/ EFT #:   000002950440',{'entities': [(48, 60, 'check_number'),(22, 29, 'check_instruction')]}),    
('Check team Pls VOID Check #3018677?',{'entities': [(27, 34, 'check_number'),(15, 19, 'check_instruction')]}),
('Please stop pay check Vehicle is a total loss Check #/ EFT #: 000002861786 ',{'entities': [(62, 74, 'check_number'),(7, 11, 'check_instruction')]}),
('Pls stop pymt and reissue.; Pls overnight, no ignature required.; Thx; Check #/ EFT #: 000002989352',{'entities': [(87, 99, 'check_number'),(18, 25, 'check_instruction')]}),
('Please verify if check has cleared, if not please stop pay on check, I will reissue overnight mail , please send me task to issue check overnight mail after stop confirmed   ;  thank you;Check #/ EFT #: 000002968167',{'entities': [(203, 215, 'check_number'),(76, 83, 'check_instruction')]}),
('CA Assist- pls pull and void check;#/ EFT #: 000003006683',{'entities': [(45, 57, 'check_number'),(15, 19, 'check_instruction')]}),
('Please stop payment to claimant. Reissue to the address listed below.;Check #/ EFT #: 000002965680',{'entities': [(86, 98, 'check_number'),(33, 40, 'check_instruction')]}),
('please place stop pay on check;Please notify me once complete as I will need to re-issue, thank you;Check #/ EFT #: 000002968603',{'entities': [(116, 128, 'check_number'),(80, 88, 'check_instruction')]}),
('Please stop pay check 2983409 and reissue to:',{'entities': [(22, 29, 'check_number'),(34, 41, 'check_instruction')]}),
('stop pay and reissue to same address on check',{'entities': [(62, 74, 'check_number'),(13, 20, 'check_instruction')]}),
('Please stop pay and re-issue check to insd Check #/ EFT #: 000002972132',{'entities': [(59, 61, 'check_number'),(20, 28, 'check_instruction')]}),

 ('Check team Pls VOID Check #3018677?', {'entities': [(15, 19, 'check_instruction'),(27, 34, 'check_instruction')]}), 
               ('Validate Check #2983294  - Amount $24,7 ?', {'entities': [(16, 23, 'check_number')]}), 
                         ('This is not right format', {'entities': []})
 ]
    
nlp = spacy.load('en_core_web_lg')  # create language class
print(f'Current vocab size :{len(nlp.vocab)}')

# Check how it looks as default 
testTxt='Sundar Pichai Vemula works at Google. His insurance premium $2,870,820 is deducted at source. Last primium has been paid on 12/03/2019 by Check. The Check number is #2870820. This needs to Stoped/Reissued'
doc = nlp(testTxt)
for token in doc:
    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(
        token.text,
        token.idx,
        token.lemma_,
        token.is_punct,
        token.is_space,
        token.shape_,
        token.pos_,
        token.tag_
    ))
    
# Lets see how does it looks on default 
from spacy import displacy 
#displacy.serve(doc, style='ent')

ner = nlp.get_pipe('ner')

#Create custom entities 
stopNvoid=['check_number','check_instruction','check_amount']

#Add Lables into 'ner' pipe
for ent in stopNvoid:
    if 'extra_labels' in ner.cfg and ent in ner.cfg['extra_labels']:
        pass
    else:
        ner.add_label(ent)
        
ner.cfg['extra_labels']= stopNvoid

''' Need to define the training data in specific format i.e. list of tuples 
I have already prepared the data in expected format so not used but this is created for others 
Format : [ ('text to be analyzed', {'entities':[(begin,end,'entity_nm1'),(begin,end,'entity_nm2')]} ]
'''
def tagData(data,annotation = {}):
    #process your data and build list of tuples 
    taggedList=[]
    for label in annotation.get('entities'):
        taggedList.append((label[0], label[1], label[2]))
    return (data,{'entities',taggedList})



''' Your data '''
for data,annotation in GOLD_DATA:
    print(f'\n {data} \n')
    for txt in annotation.get('entities'):
        print(f'{txt[2]} \t {txt[0]} \t {txt[1]}')
        



# Adding entity recognizer if not in pipeline or use the same if present 
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)
else:
    ner = nlp.get_pipe('ner')

# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']


from spacy.util import minibatch
# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(200):
        random.shuffle(TRAIN_DATA)
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(TRAIN_DATA) #, size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
        print('Losses', losses)

test_data=['Check team Pls VOID Check #3018677?']#,'Lets stop #1234567','Stop and reissues check#009987766 of amount $450']
for data in test_data:
    doc=nlp(data)
    l=[({ent.label_:ent.text},{ent.label_:''}) for ent in doc.ents]
    for ent in doc.ents:
        #print(f'{doc.text} \n {ent.text} \t {ent.label_}' )
        pass


text = 'Sharad has taken step back and reissues a check. Check #3018677?'
doc = nlp(text)
for ent in doc.ents:
    print (ent.start_char, ent.text, ent.label_)

# Lets see how does it looks on default 
displacy.serve(doc, style='ent')
        