from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/Jocelyn

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

os.chdir('interp_obgyn_outcomes/')

# ! pip install python-Levenshtein

# import gspread
# from google.colab import auth
# auth.authenticate_user()
# from google.auth import default

# creds, _ = default()
# gc = gspread.authorize(creds)

# gsheets = gc.open_by_url('https://docs.google.com/spreadsheets/d/1_2v2023IUQo1bY7yLY9dl5yIZuoT62b2/edit#gid=1045397882')
# ## deleted this column: "Please let us know who referred you to the I Can PIC website."
# merged_data_list = gsheets.worksheet('Sheet1').get_all_values()

# prompt: extract values from csv file


import csv
insurance_list = []
with open('insurance list coded.csv', 'r') as f:
  reader = csv.reader(f)
  data = list(reader)

  # Access the values in the data list
  for i,row in enumerate(data):
    insurance_list.append(row)


import csv
merged_data_list = []
with open('combined Covid Delivery Log.csv', 'r') as f:
  reader = csv.reader(f)
  data = list(reader)

  # Access the values in the data list
  for i,row in enumerate(data):
    merged_data_list.append(row)




interpreter_list = []
with open('combined_inter_log.csv', 'r') as f:
  reader = csv.reader(f)
  data = list(reader)

  # Access the values in the data list
  for i,row in enumerate(data):
    interpreter_list.append(row)

merged_data_list = [x for x in merged_data_list[1:] if x.count('')!=123 and x[32]!='']
interp_list_name_usedInterp = [[x[0], x[10].split('/')[2].split(' ')[0].split('20')[-1], x[30]] for x in interpreter_list[1:]]




htn_list = []
with open('htn_list_annotated.csv', 'r') as f:
  reader = csv.reader(f)
  data = list(reader)

  # Access the values in the data list
  for i,row in enumerate(data):
    htn_list.append(row)



twins_list = []
for i in range(len(merged_data_list)):
  x = merged_data_list[i]
  if x[6] in ['Quads', 'Triplets', 'Twin', 'Twins', 'Twtins']:

    twins_list.append([x[0],x[31].split('/')[-1].split(' ')[0]])

# count = 0
# delivery_log_mrn_and_year_list = [[x[20],x[4].split('/')[-1]] for x in merged_data_list]
# delivery_log_MomName_and_year_list = [[x[0],x[4].split('/')[-1]] for x in merged_data_list]

# data = []

# print(f'{"Mother Name":<50} {"MRN":<10} {"Delivery Date/Time":<20}')
# print('-' * 100)

# for mother_name,MRN,delivery_date in [[x[0], x[1],x[10]] for x in interpreter_list[1:]]:
#   delivery_year = delivery_date.split('/')[2].split(' ')[0].split('20')[-1]
#   if [MRN,delivery_year] not in delivery_log_mrn_and_year_list:
#     reason = 'does not exist'
#     if [mother_name,delivery_year] in delivery_log_MomName_and_year_list:
#         reason = 'wrong MRN'

#     print(f'{mother_name:<50} {MRN:<10} {delivery_date:<20} {reason:<20}')
#     count += 1

#     data.append({
#         "Mother Name": mother_name,
#         "MRN": MRN,
#         "Delivery Date": delivery_date,
#         "Reason": reason
#     })

# # Convert the list to a DataFrame
# df = pd.DataFrame(data)

# # Export to CSV
# df.to_csv("missing_data.csv", index=False)

# df_temp = pd.DataFrame()

# for insur in np.unique([y for y in [[x[92],x[93],x[94]] for x in merged_data_list]]):
#   if len(insur)>4:
#     data_dict = {
#         "INsurance": insur
#     }
# # Append the dictionary as a new row to the DataFrame
#     df_temp = pd.concat([df_temp, pd.DataFrame([data_dict])], ignore_index=True)

# # Export to CSV
# df_temp.to_csv("insurance_list.csv", index=False)

# df_temp = pd.DataFrame()

# for hyper in np.unique([x[7] for x in merged_data_list]):
#     data_dict = {
#         "htn": hyper
#     }
# # Append the dictionary as a new row to the DataFrame
#     df_temp = pd.concat([df_temp, pd.DataFrame([data_dict])], ignore_index=True)

# # Export to CSV
# df_temp.to_csv("htn_list.csv", index=False)

def catagorize_diabetes(diabetes):
  if diabetes in ['', 'NO', 'No', 'No (prediabetes)', 'Possible', 'Pre diabetes', 'Pre-E', 'Prediabetes', 'Quads', 'Triplets', 'Twin', 'Twins', 'Twtins', 'no']:
    return 0
  elif diabetes in ['T1DM', 'T2DM', 'LADA']:
    return 1
  elif diabetes in ['GDM', 'GDM ', 'GDMA', 'GDMA Insulin controlled', 'GDMA1', 'GDMA2', 'GDMA2 v. DM2', 'Gestational Diabetes Insipidus']:
    return 2
  else:
    raise Exception('unknown diabetes type')

def catagorize_birth_type(birth_type):
  if 'C-Section' in birth_type:
    return 0
  elif birth_type in ['VBAC, Spontaneous', 'Vaginal, Spontaneous', 'Vaginal', 'Vaginal, Breech', 'Therapeutic Abortion']:
    return 1
  elif birth_type in ['Vaginal, Forceps', 'Vaginal, Vacuum (Extractor)', 'Surgical, D&E', 'vaginal, forceps']:
    return 2
  else:
    raise Exception('unknown birth type')

def catagorize_races(race):
  if race == 'American Indian or Alaska Native':
    return 0
  elif race == 'Black or African American':
    return 1
  elif race == 'White':
    return 2
  elif race == 'Asian':
    return 3
  elif race == 'Other Pacific Islander':
    return 4
  elif race == 'Unable to Answer':
    return 5
  elif race == 'Declined':
    return 6
  elif race == 'Unknown':
    return 7
  elif race == 'Other':
    return 8
  else:
    raise Exception('unknown race')

# import gspread
# from google.colab import auth
# auth.authenticate_user()
# from google.auth import default
# import time

# creds, _ = default()
# gc = gspread.authorize(creds)

# gsheets = gc.open_by_url('https://docs.google.com/spreadsheets/d/1dRsUi9P4n9IpG3zBa-SKIORQH5cGtMEF_8ewVUnHvnM/edit#gid=0')
# sheet = gsheets.sheet1

df = pd.DataFrame()

interpreter_name_list = [[x[0], x[10].split('/')[2].split(' ')[0].split('20')[-1]] for x in interpreter_list[1:]]

for i in tqdm(range(len(merged_data_list))):
  # if i>0 and i%24==0: ## gsheets write limit of 300 requests per minute
  #   time.sleep(60)

  if i==0:
    continue

    # sheet.update('A'+str(1), "patientID")
    # sheet.update('B'+str(1), "Age")
    # sheet.update('C'+str(1), "Used_interp")
    # sheet.update('D'+str(1), "Race1")
    # sheet.update('E'+str(1), "Race2")
    # sheet.update('F'+str(1), "Ethnicity")
    # sheet.update('G'+str(1), "Gravida")
    # sheet.update('H'+str(1), "Para")
    # sheet.update('I'+str(1), "GA")
    # sheet.update('J'+str(1), "Birth Weight")
    # sheet.update('K'+str(1), "Apgar 5")
    # sheet.update('L'+str(1), "Delivery Method")

  else:

    mom_name = merged_data_list[i][0]
    patientID = str(i).zfill(5)
    MomAge = merged_data_list[i][23]
    delivery_year = merged_data_list[i][31].split('/')[-1].split(' ')[0]

    if [mom_name, delivery_year] in twins_list: ## remove twin/triplets/quads entries
      continue


    ##############

    if [mom_name, delivery_year] in interpreter_name_list:
      Need_Interp = 1

      interp_result = [row for row in interp_list_name_usedInterp if row[0]==mom_name and row[1]==delivery_year]
      if len(interp_result)>0:
        Used_Interp = interp_result[0][2]
      else:
        raise Exception('no name match on interp log')
    else:
      Need_Interp = 0
      Used_Interp = 0


    ########################################
    races = merged_data_list[i][24].split('\n')
    if len(races)==1:
      Race1 = catagorize_races(races[0])
      Race2 = 6
    elif len(races)==2:
      Race1 = catagorize_races(races[0])
      Race2 = catagorize_races(races[1])
    else:
      raise Exception('three races')
    #######################################

    if "Hispanic" in merged_data_list[i][25] and "Non" not in merged_data_list[i][25]:
      Ethnicity = 1
    else:
      Ethnicity = 0

    Gravida = merged_data_list[i][28].replace('*', '')
    Para = merged_data_list[i][29].replace('*', '')

    ################### GA. ########################
    GA_val = merged_data_list[i][38]
    if GA_val == '' or GA_val == 'Unk': ## missing value
      GA = 3
    elif float(GA_val.split('w')[0])<28:
      GA= 0
    elif 28<=float(GA_val.split('w')[0])<37:
      GA = 1
    elif 37<=float(GA_val.split('w')[0]):
      GA = 2
    ###########################################

    ############### Birth Weight ############################
    BW_val = merged_data_list[i][39].replace('\t', '')
    if ' g (' not in BW_val: ## missing value or typo
      Birth_Weight = 3
    elif float(BW_val.split(' g (')[0])<1500:
      Birth_Weight = 0
    elif 1500<=float(BW_val.split(' g (')[0])<2500:
      Birth_Weight = 1
    else:
      Birth_Weight = 2
    ###########################################

    ############### Apgar 5 ############################
    APGAR5_val = merged_data_list[i][41].replace(' ', '')
    if APGAR5_val not in ['0','1','2','3','4','5','6','7','8','9','10']: ## missing or bad value
      APGAR5 = 2
    elif float(APGAR5_val)<7:
      APGAR5 = 0
    else:
      APGAR5 = 1
    ###########################################

    Delivery_Method = catagorize_birth_type(merged_data_list[i][32])

   ###################### BMI ################################
    bmi_result = [q for q in merged_data_list[i] if 'kg/m²' in q]
    if len(bmi_result)>0:
      BMI = float(bmi_result[0].split('kg/m²')[0])
      if BMI<18:
        BMI_catagory = 0 ## underweight
      elif 25>BMI>=18:
        BMI_catagory = 1 # normal
      elif 30>BMI>=25:
        BMI_catagory = 2 #overweight
      else:
        BMI_catagory = 3 # obese
    else:
      BMI_catagory = 4 # Unknown BMI
    ##########################################


    ################# HTN ##################
    htn_val = merged_data_list[i][7]

    htn_temp = [r for r in htn_list if r[0]==htn_val]
    if len(htn_temp)==1:
      HTN_intermediate = htn_temp[0][1]
    else:
      raise Exception('HTN value not recognized, or too many matches')

    if HTN_intermediate == '3':
      chronic_hypertension = 1
    else:
      chronic_hypertension = 0

    if HTN_intermediate in ['1', '2']:
      gHTN = 1
    else:
      gHTN = 0
    ###########################################

    Diabetes_type = catagorize_diabetes(merged_data_list[i][8])

    ################### EBL ###############################
    EBL_val = merged_data_list[i][16].replace('EBL ', '').replace('QBL ', '').replace('ELB ', '').replace('>', '').replace('Minimal', '0')
    EBL_val = EBL_val.replace('1000 mL + additional large volume blood loss 2/2 uterine rupture with subsequent large volume hematoma', '1200')
    EBL_val = EBL_val.replace('500 on DR; 800 on D/C note', '800')
    try:
      EBL_val = float(EBL_val)
      if EBL_val>1000:
        EBL = 1 ## hemorrhage
      else:
        EBL = 0 ## no hemorrhage
    except:
      QBL_val = merged_data_list[i][17].replace('EBL ', '').replace('QBL ', '')
      try:
        QBL_val = float(QBL_val)

        if QBL_val>1000:
          EBL = 1 ## hemorrhage
        else:
          EBL = 0 ## no hemorrhage
      except:
        EBL = 2 ## Unknown
    ######################################################

    ################ NICU ################################
    if ' to ' in merged_data_list[i][90]:
      NICU = 1
    else:
      NICU = 0
    #######################################################

    ################ Shoulder ################################
    if merged_data_list[i][75]=='Yes':
      shoulder = 1
    else:
      shoulder = 0
    #######################################################

    ################ Tear ################################
    tear_val = merged_data_list[i][72].split('\n')[0]
    if tear_val=='' or tear_val=='None':
      Tear = 0 ## no tear
    elif tear_val == '1st':
      Tear = 1
    elif tear_val == '2nd':
      Tear = 2
    elif tear_val == '3rd':
      Tear = 3
    elif tear_val == '4th':
      Tear = 4
    else:
      raise Exception('tear not known')
    #######################################################


    ################ MFM ################################
    if merged_data_list[i][9]=='MFM':
      used_MFM = 1
    else:
      used_MFM = 0
    #######################################################

    ####################################################
    insurance_value = 3 ## in case there are no matches, start with value as unknown

    insurance = merged_data_list[i][92:95] # the table be like that sometimes
    insurance = [hippo for hippo in insurance if len(hippo)>4] # filter integers
    for penguin in insurance:
      if penguin=='':
        continue
      possible_ins = [ins[1] for ins in insurance_list if penguin==ins[0]]
      if len(possible_ins)>0: # if there is at least one match
        insurance_value = int(possible_ins[0]) #take value of the first match
        break # no need to check other matches

    #####################################################


    # sheet.update('A'+str(i+1), patientID)
    # sheet.update('B'+str(i+1), MomAge)
    # sheet.update('C'+str(i+1), Used_interp)
    # sheet.update('D'+str(i+1), Race1)
    # sheet.update('E'+str(i+1), Race2)
    # sheet.update('F'+str(i+1), Ethnicity)
    # sheet.update('G'+str(i+1), Gravida)
    # sheet.update('H'+str(i+1), Para)
    # sheet.update('I'+str(i+1), GA)
    # sheet.update('J'+str(i+1), Birth_Weight)
    # sheet.update('K'+str(i+1), APGAR5)
    # sheet.update('L'+str(i+1), Delivery_Method)

    # Create a dictionary for the current row
    data_dict = {
        "patientID": patientID,
        "PatientName": mom_name,
        "Mom_Age": MomAge,
        "Race1": Race1,
        "Race2": Race2,
        "Ethnicity": Ethnicity,
        "Gravida": Gravida,
        "Para": Para,
        "Insurance": insurance_value,
        "used_MFM": used_MFM,
        "BMI": BMI_catagory,
        "chronic_HTN": chronic_hypertension,
        "Diabetes": Diabetes_type,
        # "Smoking"
        # "Other_Substance_Use"
        "gestational_HTN_disorder": gHTN,
        # Area_Deprivation_Index
        "Need_Interp": Need_Interp,
        "Used_Interpreter": Used_Interp,
        "Gestational_Age": GA,
        "Birth_Weight": Birth_Weight,
        "Delivery_Method": Delivery_Method,
        # "Length_hospital_stay"
        "EBL": EBL,
        "transferred_to_NICU": NICU,
        "shoulder_dystocia": shoulder,
        "APGAR5": APGAR5,
        "Tear": Tear
    }

    # Append the dictionary as a new row to the DataFrame
    df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)

# Export to CSV
df.to_csv("patients_data_with_name.csv", index=False)

sum(df['Insurance'].values==3)

data_sheet_list = []

for i in range(len(merged_data_list)):
  row_sheet_list = []

  if i==0:
    continue

  else:

    row_sheet_list.append(str(i).zfill(5))
    row_sheet_list.append(merged_data_list[i][3])

    if i>=435:
      row_sheet_list.append(1)
    else:
      row_sheet_list.append(0)

    ########################################
    races = merged_data_list[i][4].split('\n')
    if len(races)==1:
      row_sheet_list.append(catagorize_races(races[0]))
      row_sheet_list.append(6)
    elif len(races)==2:
      row_sheet_list.append(catagorize_races(races[0]))
      row_sheet_list.append(catagorize_races(races[1]))
    else:
      raise Exception('three races')
    #######################################

    if 'Non' in merged_data_list[i][5]:
      row_sheet_list.append(0)
    else:
      row_sheet_list.append(1)

    row_sheet_list.append(merged_data_list[i][8].replace('*', ''))
    row_sheet_list.append(merged_data_list[i][9].replace('*', ''))

    ################### GA. ########################
    if merged_data_list[i][17] == '': ## missing value
      row_sheet_list.append(3)
    elif float(merged_data_list[i][17].split('w')[0])<28:
      row_sheet_list.append(0)
    elif 28<=float(merged_data_list[i][17].split('w')[0])<37:
      row_sheet_list.append(1)
    elif 37<=float(merged_data_list[i][17].split('w')[0]):
      row_sheet_list.append(2)
    ###########################################

    ############### Birth Weight ############################
    if merged_data_list[i][18] == '': ## missing value
      row_sheet_list.append(3)
    elif float(merged_data_list[i][18].split(' g (')[0])<1500:
      row_sheet_list.append(0)
    elif 1500<=float(merged_data_list[i][18].split(' g (')[0])<2500:
      row_sheet_list.append(1)
    else:
      row_sheet_list.append(2)
    ###########################################

    ############### Apgar 5 ############################
    if merged_data_list[i][20] == '': ## missing value
      row_sheet_list.append(2)
    elif float(merged_data_list[i][20])<7:
      row_sheet_list.append(0)
    else:
      row_sheet_list.append(1)
    ###########################################

    row_sheet_list.append(catagorize_birth_type(merged_data_list[i][11]))

  data_sheet_list.append(row_sheet_list)

# prompt: python write list of lists to a csv file with a header

import csv

header = ["patientID", "Age", "Used_interp", "Race1", "Race2", "Ethnicity", "Gravida", "Para", "GA", "Birth Weight", "Apgar 5", "Delivery Method"]

with open('obgyn_data.csv', 'w', newline='') as f:
  writer = csv.writer(f)

  # write the header
  writer.writerow(header)

  # write the data
  writer.writerows(data_sheet_list)

!pwd
