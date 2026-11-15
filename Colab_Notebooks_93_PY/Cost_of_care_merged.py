from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/Jocelyn/interp_obgyb_outcomes

! pip install python-Levenshtein

import gspread
from google.colab import auth
auth.authenticate_user()
from google.auth import default

creds, _ = default()
gc = gspread.authorize(creds)

gsheets = gc.open_by_url('https://docs.google.com/spreadsheets/d/1E4VcLlr1QLt8A8kzT3SXa_-82F0RDUCAbwsVPSXWqv0/edit#gid=6079577')
## deleted this column: "Please let us know who referred you to the I Can PIC website."
data_list_intervention = gsheets.worksheet('full data set intervention arm_LK9.7.23').get_all_values()

gsheets = gc.open_by_url('https://docs.google.com/spreadsheets/d/13E78jLOrWSxn3dn1vw5jdFyVhlRT0eyM5UW9LTqgtck/edit#gid=1098080148')
## deleted this column: "Do you consent to participating in the study?"
data_list_control = gsheets.worksheet('full data set historic controls_ all cancer dx_LK9.7.23').get_all_values()



## spreadsheets are now I THINK identical up to column DG so I merged them

gsheets = gc.open_by_url('https://docs.google.com/spreadsheets/d/1rr00P33_Z24Kwy2Ax5cqqMhXKdfKzzd_hZA8EkGOCWk/edit#gid=0')
merged_data_list = gsheets.worksheet('Sheet1').get_all_values()

import numpy as np
from Levenshtein import distance as levenshtein_distance

for x in data_list_intervention[0]:
  min_dist = 10000000
  for y in data_list_control[0]:
    lev_dist = levenshtein_distance(x,y)
    if lev_dist<min_dist:
      min_dist = lev_dist

  if min_dist>0:
    print(min_dist, x)

n_tot = 0

n_cost_of_care_discussed = 0
n_insurance_eligability = 0
n_time_off_work = 0
n_cost_of_drugs = 0

n_strategies_discussed = 0

n_cost_or_strat = 0

income_cutoff = 60000
n_delayed = 0
rich_delayed = 0
poor_delayed = 0
n_rich=0
n_poor = 0
for i,x in enumerate(merged_data_list[1:137]): ## only include data points
  n_tot += 1

  # if sum(np.asarray(x[3:12]).astype('int'))>0 and x[12]!='0':
  #   print('study ID:', x[1], '--', 'date filled out:', x[2])
    # raise Exception('Did not talk about cost of care column is incorrect')

  if sum(np.asarray(x[3:12]).astype('int'))>0: ## "Were any cost topics discussed?" YES
    n_cost_of_care_discussed += 1
    if x[3]=='1':
      n_insurance_eligability += 1
    if x[9]=='1':
      n_time_off_work += 1
    if x[4]=='1':
      n_cost_of_drugs += 1

  if x[129]:
    if float(x[129])>income_cutoff:
      n_rich+=1
    else:
      n_poor += 1


  if ('1' in x[78:85]): ## if delayed
    n_delayed += 1
    if x[129]:
      if float(x[129])>income_cutoff:
        rich_delayed+=1
      else:
        poor_delayed+=1

  # if sum(np.asarray(x[14:27]).astype('int'))>0 and x[27]!='0':
  #   print('study ID:', x[1], '--', 'date filled out:', x[2])
    # raise Exception('Did not talk about stratagies column is incorrect')

  if sum(np.asarray(x[14:27]).astype('int'))>0: ## "Were any cost strategies discussed?" YES
    n_strategies_discussed += 1

  if sum(np.asarray(x[3:12]).astype('int'))>0 or sum(np.asarray(x[14:27]).astype('int'))>0:
    n_cost_or_strat += 1

round(100*poor_delayed/n_poor, 2)

round(100*rich_delayed/n_rich, 2)

print('Were any cost topics discussed?')
print('Yes:', n_cost_of_care_discussed, '(', 100*n_cost_of_care_discussed/n_tot, ')')
print('No:', n_tot-n_cost_of_care_discussed, '(', 100*(n_tot-n_cost_of_care_discussed)/n_tot, ')')
print()
print('Were any cost topics discussed?')
print('Insurance eligibility or coverage', n_insurance_eligability, '(', round(100*(n_insurance_eligability)/n_cost_of_care_discussed, 2), ')')
print('Time off work during treatment and recovery', n_time_off_work, '(', round(100*(n_time_off_work)/n_cost_of_care_discussed, 2), ')')
print('Cost of drugs to treat your cancer', n_cost_of_drugs, '(', round(100*(n_cost_of_drugs)/n_cost_of_care_discussed, 2), ')')
print()
print('Were any cost strategies discussed?')
print('Yes:', n_strategies_discussed, '(', round(100*n_strategies_discussed/n_tot, 2), ')')
print('No:', n_tot-n_strategies_discussed, '(', round(100*(n_tot-n_strategies_discussed)/n_tot, 2), ')')

n_cost_or_strat

!wget 'https://github.com/google/fonts/tree/main/apache/calligraffitti/Calligraffitti-Regular.ttf'

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

# Save the chart so we can loop through the bars below.
bars = ax.bar(
    x=np.arange(2),
    height=[62.5, 37.5],
    tick_label=['Yes', 'No']
)

# Axis formatting.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

# Add text annotations to the top of the bars.
bar_color = bars[0].get_facecolor()
for bar in bars:
  ax.text(
      bar.get_x() + bar.get_width() / 2,
      bar.get_height() + 0.3,
      round(bar.get_height(), 1),
      horizontalalignment='center',
      color=bar_color,
      weight='bold'
  )

# Add labels and a title.
ax.set_xlabel('Was Cost Discussed?', labelpad=15, color='#333333')
ax.set_ylabel('Percent of Conversations', labelpad=15, color='#333333')
ax.set_title('Percent of Patient-Physician Conversations Discussing Cost', pad=15, color='#333333',
             weight='bold')

fig.tight_layout()

plt.savefig('figures/Cost_discussion_graph.png', bbox_inches='tight', dpi=300)

!pwd

import matplotlib.pyplot as plt

# Sample data
categories = ['Yes', 'No']
values = [62.5, 37.5]

fig, ax = plt.subplots()

# Plot shadow bars (darker color)
ax.bar(categories, values, color='grey', alpha=0.3)

# Plot main bars
ax.bar(categories, values, color='blue')

# Add text annotations to the top of the bars
for i, value in enumerate(values):
    ax.text(i, value + 2, f'{value}%', ha='center', va='center', color='black')

# Add labels and a title
ax.set_xlabel('Was Cost Discussed?', labelpad=15, color='#333333')
ax.set_ylabel('Percent of Conversations', labelpad=15, color='#333333')
ax.set_title('Percent of Patient-Physician Conversations that Discussed Cost', pad=15, color='#333333', weight='bold')

plt.show()
