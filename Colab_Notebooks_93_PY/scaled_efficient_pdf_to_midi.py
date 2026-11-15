import numpy as np
%cd /content
!rm -rf temp
!mkdir temp
%cd temp

!which python # should return /usr/local/bin/python
!python --version
!echo $PYTHONPATH
%env PYTHONPATH=

%%bash
MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.11-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX &> /dev/null

!which conda # should return /usr/local/bin/conda
!conda --version # should return 4.5.11
!which python # still returns /usr/local/bin/python

%%bash
conda install --channel defaults conda python=3.7 --yes &> /dev/null
conda update --channel defaults --all --yes &> /dev/null

!conda --version # now returns 4.8.3
!python --version # now returns Python 3.6.10 :: Anaconda, Inc.
import sys
sys.path

_ = (sys.path
        .append("/usr/local/lib/python3.7/site-packages"))

!conda install -yc conda-forge opencv &> /dev/null
!conda install -yc conda-forge pdf2image &> /dev/null
!conda install -yc conda-forge poppler &> /dev/null
!conda install -y matplotlib &> /dev/null
!conda install -yc conda-forge pretty_midi &> /dev/null

# import sys
# sys.path

# _ = (sys.path
#         .append("/usr/local/lib/python3.7/site-packages"))

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/pdf_to_midi

from pdf2image import convert_from_path
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
from PIL import Image
import glob
import pretty_midi
from pretty_midi.utilities import note_number_to_name, note_name_to_number

BASE_PATH = './'

# # replace with cv2 images to avooid annoying warning
# for im_path in glob.glob('templates/**/*.png', recursive=True):
#     image = cv2.imread(im_path, cv2.IMREAD_COLOR)
#     cv2.imwrite(im_path,image)

pdf_path = 'UG_midi/PDFs/HOMESICK OFFICIAL by Dua Lipa @ Ultimate-Guitar.Com.pdf'
key = ['F#', 'major']
bpm = 95
starting_time_sig = 'four_four' # four_four, two_four
scale = 4 # downscale by this factor

template_list = [
                ['extension', BASE_PATH + 'templates/extended_notes/up.png',0.5],
                ['clef', BASE_PATH + 'templates/clef.png',0.9],
                ['measure_bar', BASE_PATH + 'templates/measure_bars/measure_bar_filtered.png',0.8],
                ['triplet_3_straight', BASE_PATH + 'templates/triplet/3_straight.png',0.8],
                ['triplet_3_tilted_right', BASE_PATH + 'templates/triplet/3_tilted_right.png',0.8],
                ['left_edge_down_above', BASE_PATH + 'templates/triplet/above/left_edge_down.png',0.9],
                ['right_edge_up_above', BASE_PATH + 'templates/triplet/above/right_edge_up.png',0.9],
                ['left_edge_straight_below', BASE_PATH + 'templates/triplet/below/left_edge_straight.png',0.9],
                ['left_edge_down_below', BASE_PATH + 'templates/triplet/below/left_edge_down.png',0.9],
                ['left_edge_up_below', BASE_PATH + 'templates/triplet/below/left_edge_up.png',0.9],
                ['right_edge_straight_below', BASE_PATH + 'templates/triplet/below/right_edge_straight.png',0.9],
                ['right_edge_down_below', BASE_PATH + 'templates/triplet/below/right_edge_down.png',0.9],
                ['right_edge_up_below', BASE_PATH + 'templates/triplet/below/right_edge_up.png',0.9],
                ['dark_note_tail_down', BASE_PATH + 'templates/dark_note_tail_down.png',0.8],
                ['tie_up_left', BASE_PATH + 'templates/tie_up_left.png',0.8],
                ['tie_down_left', BASE_PATH + 'templates/tie_down_left.png',0.8],
                ['two_four', BASE_PATH + 'templates/two_four.png',0.9],
                ['three_four', BASE_PATH + 'templates/three_four.png',0.9],
                ['four_four', BASE_PATH + 'templates/four_four.png',0.9],
                ['five_four', BASE_PATH + 'templates/five_four.png',0.9],
                ['note_head_dark', BASE_PATH + 'templates/note_head_dark.png',0.8],
                ['note_head_dark_X', BASE_PATH + 'templates/note_head_dark_X.png',0.8],
                ['dot', BASE_PATH + 'templates/dot.png',0.9],
                ['double_dot', BASE_PATH + 'templates/double_dot.png',0.88],
                ['half_note', BASE_PATH + 'templates/half_note.png',0.8],
                ['whole_note', BASE_PATH + 'templates/whole_note.png',0.85],
                ['whole_rest', BASE_PATH + 'templates/whole_rest.png',0.97],
                ['half_rest', BASE_PATH + 'templates/half_rest.png',0.97],
                ['quarter_rest', BASE_PATH + 'templates/quarter_rest.png',0.97],
                ['eigth_rest', BASE_PATH + 'templates/eigth_rest.png',0.9],
                ['sixteenth_rest', BASE_PATH + 'templates/sixteenth_rest.png',0.97],
                ['thirtysecond_rest', BASE_PATH + 'templates/thirtysecond_rest.png',0.9],
                ['eight_straight_left_up', BASE_PATH + 'templates/eight_notes/straight_left_up.png',0.9],
                ['eight_straight_right_down', BASE_PATH + 'templates/eight_notes/straight_right_down.png',0.9],
                ['eight_straight_left_down', BASE_PATH + 'templates/eight_notes/straight_left_down.png',0.9],
                ['eight_slantedDown_right_down', BASE_PATH + 'templates/eight_notes/slantedDown_right_down.png',0.9],
                ['eight_slantedDown_right_up', BASE_PATH + 'templates/eight_notes/slantedDown_right_up.png',0.9],
                ['eight_slantedDown_right_up2', BASE_PATH + 'templates/eight_notes/slantedDown_right_up2.png',0.9],
                ['eight_slantedDown_left_up', BASE_PATH + 'templates/eight_notes/slantedDown_left_up.png',0.9],
                ['eight_slantedDown_left_down', BASE_PATH + 'templates/eight_notes/slantedDown_left_down.png',0.9],
                ['eight_slantedUp_left_down', BASE_PATH + 'templates/eight_notes/slantedUp_left_down.png',0.9],
                ['eight_slantedUp_left_down2', BASE_PATH + 'templates/eight_notes/slantedUp_left_down2.png',0.85],
                ['eight_slantedUp_left_up', BASE_PATH + 'templates/eight_notes/slantedUp_left_up.png',0.9],
                ['eight_slantedUp_right_up', BASE_PATH + 'templates/eight_notes/slantedUp_right_up.png',0.9],
                ['eight_slantedUp_right_up2', BASE_PATH + 'templates/eight_notes/slantedUp_right_up2.png',0.9],
                ['eight_slantedUp_right_down', BASE_PATH + 'templates/eight_notes/slantedUp_right_down.png',0.9],
                ['eight_slantedUp_right_down2', BASE_PATH + 'templates/eight_notes/slantedUp_right_down2.png',0.9],
                ['sixteenth_slantedDown_right_up', BASE_PATH + 'templates/sixteenth_notes/slantedDown_right_up.png',0.92],
                ['sixteenth_slantedUp_right_down', BASE_PATH + 'templates/sixteenth_notes/slantedUp_right_down.png',0.9],
                ['sixteenth_slantedUp_right_down2', BASE_PATH + 'templates/sixteenth_notes/slantedUp_right_down2.png',0.9],
                ['sixteenth_slantedUp_right_up', BASE_PATH + 'templates/sixteenth_notes/slantedUp_right_up.png',0.9],
                ['sixteenth_slantedDown_right_down', BASE_PATH + 'templates/sixteenth_notes/slantedDown_right_down.png',0.9],
                ['sixteenth_slantedDown_left_down', BASE_PATH + 'templates/sixteenth_notes/slantedDown_left_down.png',0.9],
                ['sixteenth_slantedDown_left_down2', BASE_PATH + 'templates/sixteenth_notes/slantedDown_left_down2.png',0.9],
                ['sixteenth_slantedUp_left_up', BASE_PATH + 'templates/sixteenth_notes/slantedUp_left_up.png',0.95],
                ['sixteenth_slantedUp_left_down', BASE_PATH + 'templates/sixteenth_notes/slantedUp_left_down.png',0.95],
                ['sixteenth_slantedUp_left_down2', BASE_PATH + 'templates/sixteenth_notes/slantedUp_left_down2.png',0.95],
                ['sixteenth_straight_right_down', BASE_PATH + 'templates/sixteenth_notes/straight_connected_right_down.png',0.9],
                ['sixteenth_straight_left_down', BASE_PATH + 'templates/sixteenth_notes/straight_connected_left_down.png',0.9],
                ['sixteenth_straight_left_up', BASE_PATH + 'templates/sixteenth_notes/straight_left_up.png',0.9],
                ['thirtysecond_straight_right_down', BASE_PATH + 'templates/thirtysecond_notes/straight_right_down.png',0.9],
                ['thirtysecond_straight_left_down', BASE_PATH + 'templates/thirtysecond_notes/straight_left_down.png',0.9],
                ['thirtysecond_slantedUp_right_down', BASE_PATH + 'templates/thirtysecond_notes/slantedUp_right_down.png',0.9],
                ['thirtysecond_flag_down', BASE_PATH + 'templates/thirtysecond_flag_down.png',0.7],
                ['sixteenth_flag_down', BASE_PATH + 'templates/sixteenth_flag_down.png',0.7],
                ['sixteenth_flag_up', BASE_PATH + 'templates/sixteenth_flag_up.png',0.75],
                ['eigth_flag_down', BASE_PATH + 'templates/eigth_flag_down.png',0.65],
                ['eigth_flag_up', BASE_PATH + 'templates/eigth_flag_up.png',0.7],
                ['sharp', BASE_PATH + 'templates/sharp.png',0.8],
                ['natural', BASE_PATH + 'templates/natural.png',0.8],
                ['flat', BASE_PATH + 'templates/flat.png',0.8],
                ]

template_dic = {}
for x in template_list:
  temp = cv2.imread(x[1], cv2.IMREAD_COLOR)
  resized = cv2.resize(temp, (round(temp.shape[1]/scale), round(temp.shape[0]/scale)), interpolation = cv2.INTER_AREA)
  template_dic[x[0]] = [np.array(resized), x[2]]

pitch_ref = [
    ['E7', 227],
    ['D7', 215],
    ['C7', 202],
    ['B6', 188],
    ['A6', 176],
    ['G6', 164],
    ['F6', 150],
    ['E6', 136],
    ['D6', 125],
    ['C6', 110],
    ['B5', 96],
    ['A5', 82],
    ['G5', 69],
    ['F5', 56],
    ['E5', 43],
    ['D5', 28],
    ['C5', 16],
    ['B4', 2],
    ['A4', -10],
    ['G4', -23],
    ['F4', -37],
    ['E4', -51],
    ['D4', -62],
    ['C4', -76],
    ['B3', -88],
    ['A3', -103],
    ['G3', -116],
    ['F3', -129],
]
pitch_ref = [[x[0], round(x[1]/scale)] for x in pitch_ref]

for file in glob.glob('temp_pdf_pages/*.jpg'):
    os.remove(file)

# real resolution: 300/4.16667
for i,im in enumerate(convert_from_path(pdf_path, dpi=300)):
    im = np.array(im)
    Image.fromarray(im).save('temp.png')
    temp = cv2.imread('temp.png', cv2.IMREAD_COLOR)
    print(temp.shape)
    print((round(temp.shape[0]/scale), round(temp.shape[1]/scale)))
    temp = cv2.resize(temp, (round(temp.shape[1]/scale), round(temp.shape[0]/scale)), interpolation = cv2.INTER_AREA)
    print(temp.shape)
    filename = 'temp_pdf_pages/' + 'im' + str(i).zfill(3) + '.jpg'
    Path(filename).touch()
    cv2.imwrite(filename, temp)

from IPython.display import Image as im_display

im_display('temp.png')

def get_dim(symbol):
    template = template_dic[symbol][0]
    h = template.shape[0]
    w = template.shape[1]
    return h,w

def get_symbol_centers(jpg_path, symbol, dist_threshold=2):

    dist_threshold = dist_threshold/scale

    threshold = float(template_dic[symbol][1])
    template = template_dic[symbol][0]
    h = template.shape[0]
    w = template.shape[1]

    res = cv2.matchTemplate(img_rgb_clean,template,cv2.TM_CCOEFF_NORMED)

    loc = np.where( res >= threshold)

    pt_list=[]
    for pt in zip(*loc[::-1]):
        pt_list.append(pt)
    pt_list = np.asarray(pt_list)

    pt_list_filtered = []
    for pt in pt_list:
        if len(pt_list_filtered)==0:
            pt_list_filtered.append(pt)
        else:
            if np.linalg.norm(pt-pt_list_filtered, axis=1).min()>dist_threshold:
                pt_list_filtered.append(pt)

    return pt_list_filtered

def rgb_to_bgr(color):
    return (int(color[2]), int(color[1]), int(color[0]))

def get_accidentals(key):
    # KEY FORMAT; ['C#', 'minor']

    major_sharp_keys = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#']
    minor_sharp_keys = ['A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#']
    sharps_ref = ['F','C','G','D','A','E','B']

    major_flat_keys = ['C', 'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb']
    minor_flat_keys = ['A', 'D', 'G', 'C', 'F', 'Bb', 'Eb', 'Ab']
    flats_ref = ['B', 'E', 'A', 'D', 'G', 'C', 'F']

    if key[1]=='major':
        if key[0] in major_sharp_keys:
            n_sharps = [i for i,x in enumerate(major_sharp_keys) if x==key[0]][0]
            acc_type = 'sharps'
            accidentals = sharps_ref[:n_sharps]
        elif key[0] in major_flat_keys:
            n_flats = [i for i,x in enumerate(major_flat_keys) if x==key[0]][0]
            acc_type = 'flats'
            accidentals = flats_ref[:n_flats]
        else:
            raise Exception('key not found')
    elif key[1]=='minor':
        if key[0] in minor_sharp_keys:
            n_sharps = [i for i,x in enumerate(minor_sharp_keys) if x==key[0]][0]
            acc_type = 'sharps'
            accidentals = sharps_ref[:n_sharps]
        elif key[0] in minor_flat_keys:
            n_flats = [i for i,x in enumerate(minor_flat_keys) if x==key[0]][0]
            acc_type = 'flats'
            accidentals = flats_ref[:n_flats]   
        else:
            raise Exception('key not found')
    else:
        raise Exception('name error')
        
    return accidentals, acc_type

def find_extension(note):
    # returns 0 if match below (note belonging to clef below)
    # returns 1 if match above (note belonging to clef above)

    h_note_head, w_note_head = get_dim('note_head_dark')

    # rectangle below?
    h1 = int(note[1]+h_note_head)
    h2 = int(note[1]+h_note_head+(90/scale))
    w1 = int(note[0])
    w2 = int(note[0]+w_note_head)

    # rectangle above?
    w1a = int(note[0])
    w2a = int(note[0]+w_note_head)
    h1a = int(note[1]-(90/scale))
    h2a = int(note[1])

    template = template_dic['extension']

    below_match = (cv2.matchTemplate(img_rgb_clean[h1:h2,w1:w2,:],template,cv2.TM_CCOEFF_NORMED)>0.4).sum()
    above_match = (cv2.matchTemplate(img_rgb_clean[h1a:h2a,w1a:w2a,:],template,cv2.TM_CCOEFF_NORMED)>0.4).sum()

    if below_match>0:
        if above_match>0:
            cv2.putText(img_rgb, 'this', (int(note[0]), int(note[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, red,10)
            raise Exception('match above and below')
        else:
            match=0 # only match below
    else:
        if above_match>0:
            match=1 # only match above
        else:
            cv2.putText(img_rgb, 'this', (int(note[0]), int(note[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, red,10)
            raise Exception('no match above nor below')

    return match

def assign_clef_to_notes(note_pts, note_type):
    notes_with_clef_ind = []
    clef_inds = list(range(clef_pts.shape[0]))
    h,w = get_dim('clef')
    for note in note_pts:
        clef_num=99 # initialize to 99 to know when algo messed up
        banana=0
        for i in clef_inds:
            if (clef_pts[i][1]-(50/scale))<=note[1]<=(clef_pts[i][1]+h+(50/scale)):
#                 cv2.putText(img_rgb, str(i), (int(note[0]), int(note[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, aqua,10)
                clef_num=i
                banana=1 # clef found, don't look for extension
   

        if not banana:    # clef not found, look for extension  
            ext = find_extension(note)
            if ext==0:
                clef_num = np.where(np.sort(np.hstack((note[1],clef_pts[:,1])))==note[1])[0][0]
#                 cv2.putText(img_rgb, str(clef_num), (int(note[0]), int(note[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, aqua,10)
            elif ext==1:
                clef_num = np.where(np.sort(np.hstack((note[1],clef_pts[:,1])))==note[1])[0][0]-1
#                 cv2.putText(img_rgb, str(clef_num), (int(note[0]), int(note[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, aqua,10)

        measure_num = [q for q,x in enumerate(measures[clef_num]) if x[0]<=note[0]<=x[1]]
        if len(measure_num)==1:
            measure_num=measure_num[0]
        else:
            cv2.putText(img_rgb, 'here', (int(note[0]), int(note[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, aqua, 2);
            raise Exception('more than one measure match for note')
        
        notes_with_clef_ind.append((note[0], note[1], clef_num, measure_num, note_type))
    return notes_with_clef_ind

def dots_near_note(note,symbol):
    if len(dot_pts)==0:
        return []
    
    h,w = get_dim(symbol)
    temp = np.asarray(dot_pts)

    a = np.logical_and((note[0])<=temp[:,0], temp[:,0]<=(note[0]+w+(20/scale))) 
    b = np.logical_and(a, note[1]<=temp[:,1])
    c = np.logical_and(b, temp[:,1]<=(note[1]+h))

    return [dot_pts[i] for i,x in enumerate(c) if x]

def double_dots_near_note(note, symbol):
    if len(double_dot_pts)==0:
        return []
    
    h,w = get_dim(symbol)
    temp = np.asarray(double_dot_pts)

    a = np.logical_and((note[0])<=temp[:,0], temp[:,0]<=(note[0]+w+(20/scale))) 
    b = np.logical_and(a, note[1]<=temp[:,1])
    c = np.logical_and(b, temp[:,1]<=(note[1]+h))

    return [double_dot_pts[i] for i,x in enumerate(c) if x]

def accidental_near_note(accidental_list, note, symbol):
    if len(accidental_list)==0:
        return []
    
    h,w = get_dim(symbol)
    temp = np.asarray(accidental_list)

    a = np.logical_and((note[0])>=temp[:,0], (temp[:,0]+w+(20/scale))>=note[0]) 
    b = np.logical_and(a, note[1]>=temp[:,1])
    c = np.logical_and(b, note[1]<=(temp[:,1]+h))

    return [accidental_list[i] for i,x in enumerate(c) if x]

def value_dark_note(note):
    
    center_w, center_h = get_note_center(note, 'note_head_dark')
    clef_h = get_note_center(clef_pts[note[2]], 'clef')[1]
    h_diff = clef_h - center_h
    
    h_note_head, w_note_head = get_dim('note_head_dark')

    flags_for_dark_note = []
    if len(all_dark_note_flag_pts)>0 and h_diff>(-100/scale): # if note is far below clef, don't get flags under it
        ####################### get all flags under dark head note #############
        temp = np.asarray([x[:-1] for x in all_dark_note_flag_pts])

        a = np.logical_and((note[0]-(20/scale))<=temp[:,0], temp[:,0]<=(note[0]+(20/scale))) # left edge
        b = np.logical_and((note[0]-(20/scale))<=temp[:,3], temp[:,3]<=(note[0]+(20/scale)))  # right edge
        c = np.logical_or(a,b)
        d = np.logical_and(c, note[1]<=temp[:,2])
        e = np.logical_and(d, temp[:,2]<=(note[1]+(250/scale)))

        flags_for_dark_note += [all_dark_note_flag_pts[i] for i,x in enumerate(e) if x]
        #####################################################################

    if len(all_dark_note_flag_pts)>0 and h_diff<(100/scale): # if note is far above clef, don't get flags above it
        ####################### get all flags above dark head note #############
        temp = np.asarray([x[:-1] for x in all_dark_note_flag_pts])

        a = np.logical_and((note[0]+w_note_head-(20/scale))<=temp[:,0], temp[:,0]<=(note[0]+w_note_head+(20/scale))) # left edge
        b = np.logical_and((note[0]+w_note_head-(20/scale))<=temp[:,3], temp[:,3]<=(note[0]+w_note_head+(20/scale)))  # right edge
        c = np.logical_or(a,b)
        d = np.logical_and(c, note[1]>=temp[:,2])
        e = np.logical_and(d, temp[:,2]>=(note[1]-(220/scale)))

        flags_for_dark_note += [all_dark_note_flag_pts[i] for i,x in enumerate(e) if x]
        #####################################################################

    if sum([1 for x in flags_for_dark_note if 'thirtysecond'==x[4]])>0: # its a thirtysecond note
        value = 1/32
    elif sum([1 for x in flags_for_dark_note if 'sixteenth'==x[4]])>0: # its a sixteenth note
        value = 1/16
    elif sum([1 for x in flags_for_dark_note if 'eight'==x[4]])>0: # its an eight note
        value = 1/8
    else: # it must be a quarter note
        value = 1/4

    return value

def get_note_center(note, symbol):
    h,w = get_dim(symbol)
    return note[0]+w/2, note[1]+h/2

def notes_near_tie(tie, temp_note_list):
    # returns index of the note within a box of (80+w,80) pixels around the tie
    # (see cv2.rectangle in tie_pts definition for visualization)
    # if there are multiple points in box, the one with the center closest to the tie center is returned
    temp = np.asarray([(x[4], x[5]) for x in temp_note_list])
    h,w = tie[5], tie[6]
    a = np.logical_and((tie[0]-(80/scale))<=temp[:,0], temp[:,0]<=(tie[0]+w))
    if tie[4]=='up':
        b = np.logical_and(a, tie[1]<=temp[:,1])
        c = np.logical_and(b, temp[:,1]<=(tie[1]+(80/scale)))
    elif tie[4]=='down':
        b = np.logical_and(a, (tie[1]+h)>=temp[:,1])
        c = np.logical_and(b, temp[:,1]>=(tie[1]+h-(80/scale)))
        
    return [(i, np.linalg.norm(np.asarray((tie[2], tie[3]))-np.asarray((temp_note_list[i][4], temp_note_list[i][5])))) for i,x in enumerate(c) if x]

def measure_bar_centers(jpg_path, symbol):

    threshold = float(template_dic[symbol][1])
    template = cv2.cvtColor(template_dic[symbol][0], cv2.COLOR_BGR2GRAY)
    
    h = template.shape[0]
    w = template.shape[1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCORR_NORMED)

    loc = np.where( res >= threshold)

    pt_list=[]
    for pt in zip(*loc[::-1]):
        pt_list.append(pt)
    pt_list = np.asarray(pt_list)

    pt_list_filtered = []
    for pt in pt_list:
        if len(pt_list_filtered)==0:
            pt_list_filtered.append(pt)
        else:
            if np.linalg.norm(pt-pt_list_filtered, axis=1).min()>(40/scale):
                pt_list_filtered.append(pt)
                
    return pt_list_filtered

## cv2 seems to use b,g,r

blue=rgb_to_bgr((0,0,255))
green=rgb_to_bgr((0,255,0))
red=rgb_to_bgr((255,0,0))
orange=rgb_to_bgr((255,165,0))
yellow=rgb_to_bgr((255,255,0))
thistle=rgb_to_bgr((216,191,216))
aqua=rgb_to_bgr((0,255,255))
pink=rgb_to_bgr((255,192,203))
silver=rgb_to_bgr((192,192,192))
purple=rgb_to_bgr((139,0,139))

min_symbol = 'blah'
min_value = 100
for symbol in [x[0] for x in template_list]:
    h,w = get_dim(symbol)
    min_temp = np.abs(h-(61/scale)) + np.abs(w-(13/scale))
    if min_temp<min_value:
        min_symbol = symbol
        min_value = min_temp
print(min_symbol, min_value)


final_note_list = []
tie_ind_list = []
current_ind = 0 # needed to keep trak of note inds across pages for tie detection
current_time_sig = starting_time_sig # initialize time signature

for jpg_path in sorted(glob.glob('./temp_pdf_pages/*.jpg')):
# for jpg_path in ['./temp_pdf_pages/im000.jpg']:
    print(jpg_path)
    img_rgb = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
    img_rgb = np.array(img_rgb[::scale, ::scale, :])
    img_rgb_clean = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
    img_rgb_clean = np.array(img_rgb_clean[::scale, ::scale, :])

    clef_pts = np.sort(get_symbol_centers(jpg_path, 'clef'))
    h,w = get_dim('clef')
    for pt in clef_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), yellow, 10)
        center_w = pt[0] + (w/2)
        center_h = pt[1] + (h/2)
        cv2.putText(img_rgb, 'O', (int(center_w), int(center_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.01, red, 4);
        for x in pitch_ref:
            cv2.putText(img_rgb, 'O', (int(center_w+(250/scale)), int(center_h-x[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.01, red, 4);

    
    ###############################################################
    low=180
    high=210
    a = np.logical_and(img_rgb_clean[:,:,0]>low, img_rgb_clean[:,:,0]<high)
    b = np.logical_and(a, img_rgb_clean[:,:,1]>low)
    c = np.logical_and(b, img_rgb_clean[:,:,1]<high)
    d = np.logical_and(c, img_rgb_clean[:,:,2]>low)
    filtered_im = np.logical_and(d, img_rgb_clean[:,:,2]<high)
    filtered_im = filtered_im*1
    filtered_im = filtered_im*255

    measure_bar_pts = []
    for clef in clef_pts[clef_pts[:,1].argsort()]:
        cut_filtered_im = filtered_im[round(clef[1]+(40/scale)):round(clef[1]+(155/scale)), :]
        cv2.imwrite('temp.png', cut_filtered_im)
        img_gray = cv2.imread('temp.png', cv2.IMREAD_GRAYSCALE)

        measure_bar_pts.append([round(2430/scale), round(clef[1]+(100/scale))]) # the final measure bar, thankfully always in the same place
        temp_list=[]
        for symbol in [x[0] for x in template_list if 'measure' in x[0]]:
#             print(symbol)
            h,w = get_dim(symbol)
            for pt in measure_bar_centers('temp.png', symbol):
                temp_list.append(pt)
        for pt in temp_list:
            if len(measure_bar_pts)==0:
                measure_bar_pts.append(pt)
            else:
                if np.linalg.norm(np.array((pt[0], clef[1]+(100/scale)))-measure_bar_pts, axis=1).min()>(50/scale):
                    measure_bar_pts.append([pt[0],round(clef[1]+(100/scale))])
    for pt in measure_bar_pts:
        cv2.putText(img_rgb, 'M', (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 10);
    ###################################################################

    five_four_pts = get_symbol_centers(jpg_path, 'five_four')
    h,w = get_dim('five_four')
    for pt in five_four_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), pink, 20)
        
    four_four_pts = get_symbol_centers(jpg_path, 'four_four')
    h,w = get_dim('four_four')
    for pt in four_four_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), pink, 20)
        
    three_four_pts = get_symbol_centers(jpg_path, 'three_four')
    h,w = get_dim('three_four')
    for pt in three_four_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), pink, 20)

    two_four_pts = get_symbol_centers(jpg_path, 'two_four')
    h,w = get_dim('two_four')
    for pt in two_four_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), pink, 10)
        
    ###############################################################
    triplet_pts = []
    temp_list=[]
    for symbol in [x[0] for x in template_list if 'triplet' in x[0]]:
#         print(symbol)
        h,w = get_dim(symbol)
        for pt in get_symbol_centers(jpg_path, symbol):
            temp_list.append(pt)
    for pt in temp_list:
        if len(triplet_pts)==0:
            triplet_pts.append(pt)
        else:
            if np.linalg.norm(pt-triplet_pts, axis=1).min()>(40/scale):
                triplet_pts.append(pt)
    for pt in triplet_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), silver, 4)
    ###################################################################
            
    left_edge_pts = []   
    for symbol in [x[0] for x in template_list if 'left_edge' in x[0]]:
#         print(symbol)
        h,w = get_dim(symbol)
        for pt in get_symbol_centers(jpg_path, symbol):
            cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), yellow, 4)
            left_edge_pts.append([pt[0], pt[1], symbol])

    right_edge_pts = []
    for symbol in [x[0] for x in template_list if 'right_edge' in x[0]]:
#         print(symbol)
        h,w = get_dim(symbol)
        for pt in get_symbol_centers(jpg_path, symbol):
            cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), yellow, 4)
            right_edge_pts.append([pt[0], pt[1], symbol])
    
            
    dark_note_head_pts = []
    for symbol in [x[0] for x in template_list if 'note_head_dark' in x[0]]:
#         print(symbol)
        h,w = get_dim(symbol)
        for pt in get_symbol_centers(jpg_path, symbol, dist_threshold=30):
            cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), blue, 2)
            dark_note_head_pts.append(pt)
            



    all_dark_note_flag_pts = [] ## get average on left and right edges
    for symbol in [x[0] for x in template_list if 'eight' in x[0]]:
#         print(symbol)
        h,w = get_dim(symbol)
        for pt in get_symbol_centers(jpg_path, symbol):
            cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), green, 2)
            # left_w, top_h, edge_avg_h, right_w
            all_dark_note_flag_pts.append((pt[0], pt[1], pt[1] + h/2, pt[0] + w, 'eight'))


    for symbol in [x[0] for x in template_list if 'sixteenth_s' in x[0]]:
#         print(symbol)
        h,w = get_dim(symbol)
        for pt in get_symbol_centers(jpg_path, symbol):
            cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), red, 2)
            all_dark_note_flag_pts.append((pt[0], pt[1], pt[1] + h/2, pt[0] + w, 'sixteenth'))
            
            
    for symbol in [x[0] for x in template_list if 'thirtysecond_s' in x[0]]:
#         print(symbol)
        h,w = get_dim(symbol)
        for pt in get_symbol_centers(jpg_path, symbol):
            cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), purple, 2)
            all_dark_note_flag_pts.append((pt[0], pt[1], pt[1] + h/2, pt[0] + w, 'thirtysecond'))


    whole_rest_pts = get_symbol_centers(jpg_path, 'whole_rest')
    h,w = get_dim('whole_rest')
    for pt in whole_rest_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), orange, 10)

    half_rest_pts = get_symbol_centers(jpg_path, 'half_rest')
    h,w = get_dim('half_rest')
    for pt in half_rest_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), orange, 2)

    quarter_rest_pts = get_symbol_centers(jpg_path, 'quarter_rest')
    h,w = get_dim('quarter_rest')
    for pt in quarter_rest_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), orange, 5)

    eigth_rest_pts = get_symbol_centers(jpg_path, 'eigth_rest')
    h,w = get_dim('eigth_rest')
    for pt in eigth_rest_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), aqua, 10)

    sixteenth_rest_pts = get_symbol_centers(jpg_path, 'sixteenth_rest')
    h,w = get_dim('sixteenth_rest')
    for pt in sixteenth_rest_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), aqua, 2)

    thirtysecond_rest_pts = get_symbol_centers(jpg_path, 'thirtysecond_rest')
    h,w = get_dim('thirtysecond_rest')
    for pt in thirtysecond_rest_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), orange, 10)
        
    half_note_head_pts = get_symbol_centers(jpg_path, 'half_note')
    h,w = get_dim('half_note')
    for pt in half_note_head_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), (100,0,100), 2)

    whole_note_head_pts = get_symbol_centers(jpg_path, 'whole_note')
    h,w = get_dim('whole_note')
    for pt in whole_note_head_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), (10,100,255), 2)

    dot_pts = get_symbol_centers(jpg_path, 'dot')
    h,w = get_dim('dot')
    for pt in dot_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), (100,100,0), 2)
        
    double_dot_pts = get_symbol_centers(jpg_path, 'double_dot')
    h,w = get_dim('double_dot')
    for pt in double_dot_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), (100,100,0), 2)

    sharp_pts = get_symbol_centers(jpg_path, 'sharp')
    h,w = get_dim('sharp')
    for pt in sharp_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), silver, 2)
        
    natural_pts = get_symbol_centers(jpg_path, 'natural')
    h,w = get_dim('natural')
    for pt in natural_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), green, 2)
        
    flat_pts = get_symbol_centers(jpg_path, 'flat')
    h,w = get_dim('flat')
    for pt in flat_pts:
        cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), silver, 2)
        
    for symbol in [x[0] for x in template_list if 'eigth_flag' in x[0]]:
#         print(symbol)
        h,w = get_dim(symbol)
        for pt in get_symbol_centers(jpg_path, symbol):
            cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), green, 10)
            all_dark_note_flag_pts.append((pt[0], pt[1], pt[1] + h/2, pt[0] + w, 'eight'))

    for symbol in [x[0] for x in template_list if 'sixteenth_flag' in x[0]]:
#         print(symbol)
        h,w = get_dim(symbol)
        for pt in get_symbol_centers(jpg_path, symbol):
            cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), red, 10)
            all_dark_note_flag_pts.append((pt[0], pt[1], pt[1] + h/2, pt[0] + w, 'sixteenth'))

            
    for symbol in [x[0] for x in template_list if 'thirtysecond_flag' in x[0]]:
#         print(symbol)
        h,w = get_dim(symbol)
        for pt in get_symbol_centers(jpg_path, symbol):
            cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), thistle, 10)
            all_dark_note_flag_pts.append((pt[0], pt[1], pt[1] + h/2, pt[0] + w, 'thirtysecond'))
            
            
    ###############################################################
    tie_pts=[]
    for symbol in [x[0] for x in template_list if 'tie_up' in x[0]]:
#         print(symbol)
        h,w = get_dim(symbol)
        for pt in get_symbol_centers(jpg_path, symbol):
            cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), orange, 5)
            cv2.rectangle(img_rgb, (pt[0]-round(80/scale), pt[1]), (pt[0] + w, pt[1] + round(80/scale)), thistle, 3)
            tie_pts.append((pt[0], pt[1], pt[0]+w/2, pt[1]+h/2, 'up', h, w))
            
    for symbol in [x[0] for x in template_list if 'tie_down' in x[0]]:
#         print(symbol)
        h,w = get_dim(symbol)
        for pt in get_symbol_centers(jpg_path, symbol):
            cv2.rectangle(img_rgb, (int(pt[0]), int(pt[1])), (pt[0] + w, pt[1] + h), orange, 5)
            cv2.rectangle(img_rgb, (pt[0]-round(80/scale), pt[1]+h), (pt[0] + w, pt[1]+h-round(80/scale)), thistle, 3)
            tie_pts.append((pt[0], pt[1], pt[0]+w/2, pt[1]+h/2, 'down', h, w))
    ################################################################
    
    cv2.imwrite('res.png', img_rgb)

    ## clef[1] is the top of the clef box
    ## clef[1]+h is the bottom of the clef box

    measure_num=0
    measures = []
    for clef in clef_pts:
        h,w = get_dim('clef')

        measure_bars = np.sort([clef] + [x for x in measure_bar_pts if (clef[1]-h)<x[1]<(clef[1]+h)], axis=0)
        measures.append(np.vstack((measure_bars[0:-1,0],measure_bars[1:,0])).T)

    #     for i in measures:
    #         cv2.putText(img_rgb, str(measure_num), (int(i.mean()), int(clef[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, red,10)
    #         measure_num+=1

    all_notes = []
    all_notes += assign_clef_to_notes(whole_note_head_pts, 'whole_note_head_pts')
    all_notes += assign_clef_to_notes(dark_note_head_pts, 'dark_note_head_pts')
    all_notes += assign_clef_to_notes(half_note_head_pts, 'half_note_head_pts')

    all_notes += assign_clef_to_notes(thirtysecond_rest_pts, 'thirtysecond_rest_pts')
    all_notes += assign_clef_to_notes(sixteenth_rest_pts, 'sixteenth_rest_pts')
    all_notes += assign_clef_to_notes(eigth_rest_pts, 'eigth_rest_pts')
    all_notes += assign_clef_to_notes(quarter_rest_pts, 'quarter_rest_pts')
    all_notes += assign_clef_to_notes(half_rest_pts, 'half_rest_pts')
    all_notes += assign_clef_to_notes(whole_rest_pts, 'whole_rest_pts')


    all_notes_sorted = sorted(all_notes,key=lambda x: (x[2],x[0]))

    accidentals, acc_type = get_accidentals(key)

    
    temp_note_list = []
    current_measure = all_notes_sorted[0][3]
    accidental_dictionary = {}
    for note in all_notes_sorted:
        if note[3]!=current_measure: # clear accidental list
            current_measure = note[3]
            accidental_dictionary = {}
            
        center_w, center_h = 'blah','blah' ## initialize to check for error
        if note[4]=='dark_note_head_pts':
            quality = 'active'
            value = value_dark_note(note)
            symbol = 'note_head_dark'
        elif note[4]=='whole_note_head_pts':
            quality = 'active'
            value = 1
            symbol = 'whole_note'
        elif note[4]=='half_note_head_pts':
            quality = 'active'
            value = 0.5
            symbol = 'half_note'
        elif note[4]=='whole_rest_pts':
            quality = 'rest'
            value = 1
            symbol = 'whole_rest'
        elif note[4]=='half_rest_pts':
            quality = 'rest'
            value = 0.5
            symbol = 'half_rest'
        elif note[4]=='quarter_rest_pts':
            quality = 'rest'
            value = 0.25
            symbol = 'quarter_rest'
        elif note[4]=='eigth_rest_pts':
            quality = 'rest'
            value = 1/8
            symbol = 'eigth_rest'
        elif note[4]=='sixteenth_rest_pts':
            quality = 'rest'
            value = 1/16
            symbol = 'sixteenth_rest'
        elif note[4]=='thirtysecond_rest_pts':
            quality = 'rest'
            value = 1/32
            symbol = 'thirtysecond_rest'
        else:
            raise Exception('note type note recognized')
            
        center_w, center_h = get_note_center(note, symbol)
        if len(dots_near_note(note,symbol))>0:
            value += value/2
            
        if len(double_dots_near_note(note,symbol))>0:
            value += value*(3/4)

        if quality=='active':
            clef_h = get_note_center(clef_pts[note[2]], 'clef')[1]
            h_diff = clef_h - center_h
            pitch = pitch_ref[np.abs(np.asarray([x[1] for x in pitch_ref])-h_diff).argmin()][0]
            if pitch[0] in accidentals:
                if acc_type=='sharps':
                    pitch = pitch[0]+'#'+pitch[1]
                elif acc_type=='flats':
                    pitch = pitch[0]+'b'+pitch[1]
    #         cv2.putText(img_rgb, 'O', (int(center_w), int(center_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.01, red, 4);
        else:
            pitch = 'rest'

        ############ Apply Accidentals ######################################
        if pitch!='rest':       
            if len(accidental_near_note(sharp_pts, note, 'sharp'))>0:
                accidental_dictionary[pitch[0]] = 'sharp'
                if pitch[1] in ['#', 'b']:
                    pitch = pitch[0]+'#'+pitch[2]
                else:
                    pitch = pitch[0]+'#'+pitch[1]
            elif len(accidental_near_note(flat_pts, note, 'flat'))>0:
                accidental_dictionary[pitch[0]] = 'flat'
                if pitch[1] in ['#', 'b']:
                    pitch = pitch[0]+'b'+pitch[2]
                else:
                    pitch = pitch[0]+'b'+pitch[1]
            elif len(accidental_near_note(natural_pts, note, 'natural'))>0:
                accidental_dictionary[pitch[0]] = 'natural'
                if pitch[1] in ['#', 'b']:
                    pitch = pitch[0]+pitch[2]
                else:
                    pitch = pitch[0]+pitch[1]
                    
            #####check if pitch set by other accidentals in measure######
            if pitch[0] in accidental_dictionary:
                if accidental_dictionary[pitch[0]]=='sharp':
                    if pitch[1] in ['#', 'b']:
                        pitch = pitch[0]+'#'+pitch[2]
                    else:
                        pitch = pitch[0]+'#'+pitch[1]
                elif accidental_dictionary[pitch[0]]=='flat':
                    if pitch[1] in ['#', 'b']:
                        pitch = pitch[0]+'b'+pitch[2]
                    else:
                        pitch = pitch[0]+'b'+pitch[1]
                elif accidental_dictionary[pitch[0]]=='natural':
                    if pitch[1] in ['#', 'b']:
                        pitch = pitch[0]+pitch[2]
                    else:
                        pitch = pitch[0]+pitch[1]
        #############################################################
        
        
        temp_note_list.append([note[2], note[3], value, quality, center_w, center_h, pitch])

    ################remove harmony notes#######################################
    # currently keeps only the highest note
    dist_thresh = (40/scale)
    harmony_pitch_inds = []
    for clef in np.unique(np.asarray([x[0] for x in temp_note_list])):
        for measure in np.unique(np.asarray([x[1] for x in temp_note_list if x[0]==clef])):
            indices = [i for i,x in enumerate(temp_note_list) if x[0]==clef and x[1]==measure]
            for i in indices:
                temp_inds = np.squeeze(np.where(np.abs(temp_note_list[i][4]-[temp_note_list[q][4] for q in indices])<dist_thresh))
                if temp_inds.ndim>0:
                    temp_inds = [indices[d] for d in temp_inds]
                    if np.std([temp_note_list[j][5] for j in temp_inds])<0.1: # heights are the same, take the rightmost one
                        melody_pitch_ind = temp_inds[np.argmax([temp_note_list[j][4] for j in temp_inds])]
                    else: # heights are different, take the highest note
                        melody_pitch_ind = temp_inds[np.argmin([temp_note_list[j][5] for j in temp_inds])]
                    harmony_pitch_inds += [x for x in temp_inds if x!=melody_pitch_ind and x not in harmony_pitch_inds]

    for i in sorted(harmony_pitch_inds, reverse=True):
        temp_note_list = [x for q,x in enumerate(temp_note_list) if q!=i]

    ############################################################################
        
        
    ####################################################################
    triplets_processed = []
    for triplet in triplet_pts:
        temp_inds = [i for i,x in enumerate(left_edge_pts) if x[0]<triplet[0]]
        temp_edge = np.asarray([[left_edge_pts[i][0],left_edge_pts[i][1]] for i in temp_inds])
        left_edge = left_edge_pts[temp_inds[np.linalg.norm(temp_edge-triplet, axis=1).argmin()]] # take the closest left edge which is on the left

        w_right_edge = get_dim('right_edge_up_below')[0] # biggest one
        temp_inds = [i for i,x in enumerate(right_edge_pts) if x[0]>triplet[0]]
        temp_edge = np.asarray([[right_edge_pts[i][0],right_edge_pts[i][1]] for i in temp_inds])
        right_edge = right_edge_pts[temp_inds[np.linalg.norm(temp_edge-triplet, axis=1).argmin()]] # take the closest right edge which is on the right

        if 'above' in right_edge[2] and 'above' in left_edge[2]:
            clef_num = np.where(np.sort(np.hstack((triplet[1],clef_pts[:,1])))==triplet[1])[0][0] # clef just below triplet
        elif 'below' in right_edge[2] and 'below' in left_edge[2]:
            clef_num = np.where(np.sort(np.hstack((triplet[1],clef_pts[:,1])))==triplet[1])[0][0]-1 # clef just above triplet
        else:
            raise Exception('left and right edges do not match')

        triplets_processed.append((left_edge[0], right_edge[0]+w_right_edge, clef_num))
        
        
    ## now apply triplets ##
    for triplet in triplets_processed:
        indices = [i for i,x in enumerate(temp_note_list) if x[0]==triplet[2] and x[4]>=triplet[0] and x[4]<=triplet[1]]
        for i in indices:
            if temp_note_list[i][2]==0.25: # quarter note goes to 1/6 value
                temp_note_list[i][2]=1/6
            elif temp_note_list[i][2]==0.125: # eigth note goes to 1/12 value
                temp_note_list[i][2]=1/12
            elif temp_note_list[i][2]==0.0625: # sixteenth note goes to 1/24 value
                temp_note_list[i][2]=1/24
    ####################################################################

    for note in temp_note_list:
        cv2.putText(img_rgb, str(note[2])[:4], (int(note[4]), int(note[5])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, aqua, 2);

    cv2.imwrite('res.png', img_rgb)

    time_sigs = assign_clef_to_notes(four_four_pts, 'four_four') 
    time_sigs += assign_clef_to_notes(three_four_pts, 'three_four') 
    time_sigs += assign_clef_to_notes(two_four_pts, 'two_four')
    time_sigs += assign_clef_to_notes(five_four_pts, 'five_four')
    for clef in np.unique(np.asarray([x[0] for x in temp_note_list])):
        for measure in np.unique(np.asarray([x[1] for x in temp_note_list if x[0]==clef])):
            time_sum = sum([x[2] for x in temp_note_list if (x[0]==clef and x[1]==measure)])
            temp_sig = [y for y in time_sigs if y[2]==clef and y[3]==measure]
            if temp_sig:
                current_time_sig = temp_sig[0][4]
                
            if current_time_sig == 'four_four':
                target_sum = 1
            elif current_time_sig == 'five_four':
                target_sum = 1.25
            elif current_time_sig == 'three_four':
                target_sum = 0.75
            elif current_time_sig == 'two_four':
                target_sum = 0.5
                
            if np.abs(time_sum-target_sum)>0.0001:
                raise Exception("measure", measure, "of clef", clef, "does not add up", 'sum is', sum([x[2] for x in temp_note_list if (x[0]==clef and x[1]==measure)]))
    print('time in all measures add up')

    final_note_list += temp_note_list

    for tie in tie_pts:
        temp = notes_near_tie(tie, temp_note_list)
        if len(temp)==0:
            continue
        elif len(temp)==1:
            ind = temp[0][0]
        else:
            ind = temp[np.asarray(temp)[:,1].argmin()][0]
#         print(ind)
#         cv2.putText(img_rgb, 'O', (int(final_note_list[ind+current_ind][4]), int(final_note_list[ind+current_ind][5])), cv2.FONT_HERSHEY_SIMPLEX, 0.01, aqua, 4);
        tie_ind_list.append(ind+current_ind)

#     cv2.imwrite('res.png', img_rgb)


    current_ind += len(temp_note_list)
    

from IPython.display import Image

cv2.imwrite('res.png', img_rgb)

Image('res.png')

# cv2.imwrite('res.png', img_rgb)
final_note_list_bak = final_note_list
# final_note_list = final_note_list_bak

## final_note_list must include all notes for all pages before running this
# (because ties may exist across pages)
for ind in sorted(tie_ind_list, reverse=True):
    cv2.putText(img_rgb, 'blah', (int(final_note_list[ind][4]), int(final_note_list[ind][5])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, blue, 4);
#     if final_note_list[ind][6]==final_note_list[ind+1][6]: # if pitch is different leave it separate
    combined_value = final_note_list[ind][2]+final_note_list[ind+1][2]
    final_note_list = [list(x) for i,x in enumerate(final_note_list) if i!=(ind+1)]
    final_note_list[ind][2] = combined_value
    

time_sig = 'four_four'
velocity = 100

if time_sig == 'four_four':
    bps = bpm/60
    sec_per_beat = 1/bps
    sec_per_measure = 4*sec_per_beat

pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
inst = pretty_midi.Instrument(program=42, is_drum=False, name='my cello')
pm.instruments.append(inst)

start = 0
for note in final_note_list:
    if note[3]=='active':
        inst.notes.append(pretty_midi.Note(velocity, note_name_to_number(note[6]), start, start+note[2]*sec_per_measure))

    start += note[2]*sec_per_measure

print(inst.notes)

# pm.write('/Users/agp2004/Desktop/UG_midi/MIDI/Dua_Lipa_homesick.mid')

    
