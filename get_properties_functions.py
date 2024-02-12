#code written by Brittany C. Haas and Melissa A. Hardy (adapted from David B. Vogt's get_properties_pandas.py, adapted from Tobias Gensch)

import pandas as pd
import numpy as np
import re
import math
from morfeus import Sterimol
from morfeus import BuriedVolume
from morfeus import Pyramidalization
from morfeus import SASA

import goodvibes.GoodVibes as gv
import goodvibes.thermo as thermo
import goodvibes.io as io

import dbstep.Dbstep as db

homo_pattern = re.compile("Alpha  occ. eigenvalues")
npa_pattern = re.compile("Summary of Natural Population Analysis:")
nbo_os_pattern = re.compile("beta spin orbitals")
nmrstart_pattern = " SCF GIAO Magnetic shielding tensor (ppm):\n"
nmrend_pattern = re.compile("End of Minotr F.D.")
nmrend_pattern_os = re.compile("g value of the free electron")
zero_pattern = re.compile("zero-point Energies")
cputime_pattern = re.compile("Job cpu time:")
walltime_pattern = re.compile("Elapsed time:")
volume_pattern = re.compile("Molar volume =")
polarizability_pattern = re.compile("Dipole polarizability, Alpha")
dipole_pattern = "Dipole moment (field-independent basis, Debye)"
frqs_pattern = re.compile("Red. masses")
frqsend_pattern = re.compile("Thermochemistry")   
chelpg1_pattern = re.compile("(CHELPG)")
chelpg2_pattern = re.compile("Charges from ESP fit")
hirshfeld_pattern = re.compile("Hirshfeld charges, spin densities, dipoles, and CM5 charges")

def get_geom(streams): #extracts the geometry from the compressed stream
    geom = []
    for item in streams[-1][16:]:
        if item == "":
            break
        geom.append([item.split(",")[0],float(item.split(",")[-3]),float(item.split(",")[-2]),float(item.split(",")[-1])])
    return(geom)

def get_outstreams(log): #gets the compressed stream information at the end of a Gaussian job
    streams = []
    starts,ends = [],[]
    error = ""
    an_error = True
    try:
        with open(log+".log") as f:
            loglines = f.readlines()
    except:
        with open(log+".LOG") as f:
            loglines = f.readlines()
            
    for line in loglines[::-1]:
        if "Normal termination" in line:
            an_error = False
        if an_error:
            error = "****Failed or incomplete jobs for " + log + ".log"        
            
    for i in range(len(loglines)):
        if "1\\1\\" in loglines[i]:
            starts.append(i)
        if "@" in loglines[i]:
            ends.append(i)
    if len(starts) != len(ends) or len(starts) == 0: #probably redundant
        error = "****Failed or incomplete jobs for " + log + ".log"
        return(streams,error)
    for i in range(len(starts)):
        tmp = ""
        for j in range(starts[i],ends[i]+1,1):
            tmp = tmp + loglines[j][1:-1]
        streams.append(tmp.split("\\"))
    return(streams,error)

def get_filecont(log): #gets the entire job output
    error = "" #default unless "normal termination" is in file
    an_error = True
    with open(log+".log") as f:
        loglines = f.readlines()
    for line in loglines[::-1]:
        if "Normal termination" in line:
            an_error = False
        if an_error:
            error = "****Failed or incomplete jobs for " + log + ".log"
    return(loglines, error)

def get_sterimol_morfeus(dataframe, sterimol_list): #uses morfeus to calculate sterimol L, B1, B5 for two input atoms for every entry in df
    sterimol_dataframe = pd.DataFrame(columns=[])
    
    for index, row in dataframe.iterrows():
        try:
            #parsing the Sterimol axis defined in the list from input line
            sterimolnums_list = [] 
            for sterimol in sterimol_list: 
                atomnum_list = [] #the atom numbers used to collect sterimol values (e.g., [18 16 17 15]) are collected from the df using the input list (e.g., [["O2", "C1"], ["O3", "H5"]])
                for atom in sterimol:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                sterimolnums_list.append(atomnum_list) #append atomnum_list for each sterimol axis defined in the input to make a list of the form [['18', '16'], ['16', '15']]
            
            #this makes column headers based on Sterimol axis defined in the input line
            sterimoltitle_list = []
            for sterimol in sterimol_list:
                sterimoltitle = str(sterimol[0]) + "_" + str(sterimol[1])
                sterimoltitle_list.append(sterimoltitle)
            
            log_file = row['log_name']
            streams, error = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(sterimolnums_list)):
                    entry = {'Sterimol_L_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': "no data", 
                    'Sterimol_B1_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': "no data", 
                    'Sterimol_B5_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': "no data"}
                    row_i.update(entry)
                sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
                continue
            
            geom = get_geom(streams)
            
                            
            #checks for if the wrong number of atoms are input, input is not of the correct form, or calls atom numbers that do not exist in the molecule
            error = ""
            for sterimol in sterimolnums_list:
                if len(sterimol)%2 != 0:
                    error = "Number of atom inputs given for Sterimol is not divisible by two. " + str(len(sterimol)) + " atoms were given. "
                for atom in sterimol:
                    if not atom.isdigit():
                        error += " " + atom + ": Only numbers accepted as input for Sterimol"
                    if int(atom) > len(geom):
                        error += " " + atom + " is out of range. Maximum valid atom number: " + str(len(geom)+1) + " "
                if error != "": print(error)
                    
            elements = np.array([geom[i][0] for i in range(len(geom))])
            coordinates = np.array([np.array(geom[i][1:]) for i in range(len(geom))])
            
            #this collects Sterimol values for each pair of inputs
            sterimolout = []
            for sterimol in sterimolnums_list:
                sterimol_values = Sterimol(elements, coordinates, int(sterimol[0]), int(sterimol[1])) #calls morfeus
                sterimolout.append(sterimol_values)
           
            
            #this adds the data from sterimolout into the new property df
            row_i = {}
            for a in range(0, len(sterimolnums_list)):
                entry = {'Sterimol_L_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': sterimolout[a].L_value, 
                'Sterimol_B1_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': sterimolout[a].B_1_value, 
                'Sterimol_B5_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': sterimolout[a].B_5_value}
                row_i.update(entry)
            sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire Morfeus Sterimol parameters for:', row['log_name'], ".log")
            row_i = {}
            try: 
                for a in range(0, len(sterimolnums_list)):
                    entry = {'Sterimol_L_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': "no data", 
                    'Sterimol_B1_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': "no data", 
                    'Sterimol_B5_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': "no data"}
                    row_i.update(entry)
                sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("Morfeus Sterimol function has completed for", sterimol_list)
    return(pd.concat([dataframe, sterimol_dataframe], axis = 1))

def get_sterimol_dbstep(dataframe, sterimol_list): #uses DBSTEP to calculate sterimol L, B1, B5 for two input atoms for every entry in df
    sterimol_dataframe = pd.DataFrame(columns=[])
    
    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']
            
            #parsing the Sterimol axis defined in the list from input line
            sterimolnums_list = [] 
            for sterimol in sterimol_list: 
                atomnum_list = [] #the atom numbers used to collect sterimol values (e.g., [18 16 17 15]) are collected from the df using the input list (e.g., [["O2", "C1"], ["O3", "H5"]])
                for atom in sterimol:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                sterimolnums_list.append(atomnum_list) #append atomnum_list for each sterimol axis defined in the input to make a list of the form [['18', '16'], ['16', '15']]
                
            #checks for if the wrong number of atoms are input or input is not of the correct form
            error = ""
            for sterimol in sterimolnums_list:
                if len(sterimol)%2 != 0:
                    error = "****Number of atom inputs given for Sterimol is not divisible by two. " + str(len(sterimol)) + " atoms were given. "
                for atom in sterimol:
                    if not atom.isdigit():
                        error += "**** " + atom + ": Only numbers accepted as input for Sterimol"
                if error != "": print(error)
            
            #this collects Sterimol values for each pair of inputs
            sterimol_out = []
            fp = log_file + str(".log")
            for sterimol in sterimolnums_list:
                sterimol_values = db.dbstep(fp,atom1=int(sterimol[0]),atom2=int(sterimol[1]),commandline=True,verbose=False,sterimol=True,measure='grid')
                sterimol_out.append(sterimol_values)
                                                            
            #this makes column headers based on Sterimol axis defined in the input line
            sterimoltitle_list = []
            for sterimol in sterimol_list:
                sterimoltitle = str(sterimol[0]) + "_" + str(sterimol[1])
                sterimoltitle_list.append(sterimoltitle)
            
            #this adds the data from sterimolout into the new property df
            row_i = {}
            for a in range(0, len(sterimolnums_list)):
                entry = {'Sterimol_B1_' + str(sterimoltitle_list[a]) + "(Å)_dbstep": sterimol_out[a].Bmin, 
                         'Sterimol_B5_' + str(sterimoltitle_list[a]) + "(Å)_dbstep": sterimol_out[a].Bmax, 
                         'Sterimol_L_' + str(sterimoltitle_list[a]) + "(Å)_dbstep": sterimol_out[a].L}
                row_i.update(entry)
            sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire DSBTEP Sterimol parameters for:', row['log_name'], ".log")
            row_i = {}
            try: 
                for a in range(0, len(sterimolnums_list)):
                    entry = {'Sterimol_L_' + str(sterimoltitle_list[a]) + '(Å)_dbstep': "no data", 
                    'Sterimol_B1_' + str(sterimoltitle_list[a]) + '(Å)_dbstep': "no data", 
                    'Sterimol_B5_' + str(sterimoltitle_list[a]) + '(Å)_dbstep': "no data"}
                    row_i.update(entry)
                sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("DBSTEP Sterimol function has completed for", sterimol_list)
    return(pd.concat([dataframe, sterimol_dataframe], axis = 1))
    
def get_sterimol2vec(dataframe, sterimol_list, end_r, step_size): #uses DBSTEP to calculate sterimol Bmin and Bmax for two input atoms at intervals from 0 to end_r at step_size
    sterimol_dataframe = pd.DataFrame(columns=[])
    num_steps = int((end_r)/step_size + 1)
    radii_list = [0 + step_size*i for i in range(num_steps)]
    
    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']
            
            #parsing the Sterimol axis defined in the list from input line
            sterimolnums_list = [] 
            for sterimol in sterimol_list: 
                atomnum_list = [] #the atom numbers used to collect sterimol values (e.g., [18 16 17 15]) are collected from the df using the input list (e.g., [["O2", "C1"], ["O3", "H5"]])
                for atom in sterimol:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                sterimolnums_list.append(atomnum_list) #append atomnum_list for each sterimol axis defined in the input to make a list of the form [['18', '16'], ['16', '15']]
                
            #checks for if the wrong number of atoms are input or input is not of the correct form
            error = ""
            for sterimol in sterimolnums_list:
                if len(sterimol)%2 != 0:
                    error = "Number of atom inputs given for Sterimol is not divisible by two. " + str(len(sterimol)) + " atoms were given. "
                for atom in sterimol:
                    if not atom.isdigit():
                        error += " " + atom + ": Only numbers accepted as input for Sterimol"
                if error != "": print(error)
            
            #this collects Sterimol values for each pair of inputs
            sterimol2vec_out = []
            fp = log_file + str(".log")
            for sterimol in sterimolnums_list:
                sterimol2vec_values = db.dbstep(fp,atom1=int(sterimol[0]),atom2=int(sterimol[1]),scan='0.0:{}:{}'.format(end_r,step_size),commandline=True,verbose=False,sterimol=True,measure='grid')
                sterimol2vec_out.append(sterimol2vec_values)
                                                            
            #this makes column headers based on Sterimol axis defined in the input line
            sterimoltitle_list = []
            for sterimol in sterimol_list:
                sterimoltitle = str(sterimol[0]) + "_" + str(sterimol[1])
                sterimoltitle_list.append(sterimoltitle)
            
            scans = radii_list
            #this adds the data from sterimolout into the new property df
            row_i = {}
            for a in range(0, len(sterimolnums_list)):
                for i in range(0, len(scans)):
                    entry = {'Sterimol_Bmin_' + str(sterimoltitle_list[a]) + "_" + str(scans[i]) + "Å(Å)": sterimol2vec_out[a].Bmin[i], 
                             'Sterimol_Bmax_' + str(sterimoltitle_list[a]) + "_" + str(scans[i]) + "Å(Å)": sterimol2vec_out[a].Bmax[i]}
                    row_i.update(entry)
            sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire DSBTEP Sterimol2Vec parameters for:', row['log_name'], ".log")
            row_i = {}
            try: 
                for a in range(0, len(sterimolnums_list)):
                    for i in range(0, len(scans)):
                        entry = {'Sterimol_Bmin_' + str(sterimoltitle_list[a]) + "_" + str(scans[i]) + "Å(Å)": "no data", 
                                'Sterimol_Bmax_' + str(sterimoltitle_list[a]) + "_" + str(scans[i]) + "Å(Å)": "no data"}
                        row_i.update(entry)
                sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")    
    print("DBSTEP Sterimol2Vec function has completed for", sterimol_list)
    return(pd.concat([dataframe, sterimol_dataframe], axis = 1))
    
def get_vbur_one_radius(dataframe, a1, radius): #uses morfeus to calculate vbur at a single radius for an atom (a1) in df
    atom = str(a1)
    vbur_dataframe = pd.DataFrame(columns=[])
    
    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']
            atom1 = row[str(a1)] #gets numerical value (e.g. 16) for a1 (e.g. C1)
            streams, error = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
            if error != "":
                print(error)
                row_i = {'%Vbur_'+str(atom)+"_"+str(radius)+"Å": "no data"}
                vbur_dataframe = vbur_dataframe.append(row_i, ignore_index=True)
                continue
            
            log_coordinates = get_geom(streams)
            elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
            coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])
            vbur = BuriedVolume(elements, coordinates, int(atom1), include_hs=True, radius=radius) #calls morfeus
            row_i = {'%Vbur_'+str(atom)+"_"+str(radius)+"Å": vbur.percent_buried_volume * 100}
            vbur_dataframe = vbur_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire Vbur parameters for:', row['log_name'], ".log")
            row_i = {'%Vbur_'+str(atom)+"_"+str(radius)+"Å": "no data"}
            vbur_dataframe = vbur_dataframe.append(row_i, ignore_index=True)
    return(vbur_dataframe)

def get_vbur_scan(dataframe, a_list, start_r, end_r, step_size): #uses morfeus via get_vbur_one_radius to scan vbur across a range of radii
    num_steps = int((end_r-start_r)/step_size + 1)
    radii = [start_r + step_size*i for i in range(num_steps)]
    frames = []
    for radius in radii:
        for a in a_list:
            frames.append(get_vbur_one_radius(dataframe, a, radius))
    vbur_scan_dataframe = pd.concat(frames, axis = 1)
    print("Vbur scan function has completed for", a_list, "from", start_r, " to ", end_r)
    return(pd.concat([dataframe, vbur_scan_dataframe], axis = 1))
    
def get_pyramidalization(dataframe, a_list): #uses morfeus to calculate pyramidalization (based on the 3 atoms in closest proximity to the defined atom) for for all atoms (a_list, of form ["C1", "C4", "O2"]) in a dataframe that contains file name and atom number
    pyr_dataframe = pd.DataFrame(columns=[])
    
    for index, row in dataframe.iterrows():
        try:
            atom_list = [] 
            for label in a_list: 
                atom = row[str(label)] #the atom number (e.g., 16) to add to the list is the df entry of this row for the labeled atom (e.g., "C1")
                atom_list.append(str(atom)) #append that to atom_list to make a list of the form [16, 17, 29]

            log_file = row['log_name']
            streams, error = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(atom_list)):
                    entry = {'pyramidalization_Gavrish_' + str(a_list[a]) + '(°)': "no data", 
                             'pyramidalization_Agranat-Radhakrishnan_' + str(a_list[a]): "no data"} #details on these values can be found here: https://kjelljorner.github.io/morfeus/pyramidalization.html
                    row_i.update(entry)
                pyr_dataframe = pyr_dataframe.append(row_i, ignore_index=True) 
                continue
            
            log_coordinates = get_geom(streams)
            elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
            coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])

            pyrout = []
            for atom in atom_list:
                pyr = Pyramidalization(coordinates, int(atom)) #calls morfeus
                pyrout.append(pyr)
        
            row_i = {}
            for a in range(0, len(atom_list)):
                entry = {'pyramidalization_Gavrish_' + str(a_list[a]) + '(°)': pyrout[a].P_angle, 
                'pyramidalization_Agranat-Radhakrishnan_' + str(a_list[a]): pyrout[a].P} #details on these values can be found here: https://kjelljorner.github.io/morfeus/pyramidalization.html
                row_i.update(entry)
            pyr_dataframe = pyr_dataframe.append(row_i, ignore_index=True)   
        except:
            print('****Unable to acquire pyramidalizataion parameters for:', row['log_name'], ".log")
            row_i = {}
            for a in range(0, len(atom_list)):
                entry = {'pyramidalization_Gavrish_' + str(a_list[a]) + '(°)': "no data", 
                'pyramidalization_Agranat-Radhakrishnan_' + str(a_list[a]): "no data"} #details on these values can be found here: https://kjelljorner.github.io/morfeus/pyramidalization.html
                row_i.update(entry)
            pyr_dataframe = pyr_dataframe.append(row_i, ignore_index=True) 
    print("Pyramidalization function has completed for", a_list)
    return(pd.concat([dataframe, pyr_dataframe], axis = 1))

def get_specdata(atoms,prop): #input a list of atom numbers and a list of pairs of all atom numbers and property of interest for use with NMR, NBO, etc.
    propout = []
    for atom in atoms:
        if atom.isdigit():
            a = int(atom)-1
            if a <= len(prop):
                propout.append(float(prop[a][1]))
            else: continue
        else: continue
    return(propout)
    
def get_nbo(dataframe, a_list): #a function to get the nbo npa partial charge for all atoms (a_list, form ["C1", "C4", "O2"]) in a dataframe that contains file name and atom number
    nbo_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in
                
    for index, row in dataframe.iterrows(): #iterate over the dataframe 
        try: #try to get the data
            atomnum_list = [] 
            for atom in a_list: 
                atomnum = row[str(atom)] #the atom number (e.g., 16) to add to the list is the df entry of this row for the labeled atom (e.g., "C1")
                atomnum_list.append(str(atomnum)) #append that to atomnum_list to make a list of the form [16, 17, 29]
            
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'NBO_charge_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                nbo_dataframe = nbo_dataframe.append(row_i, ignore_index=True)
                continue
            
            nbo,nbostart,nboout,skip = [],0,"",0 
            #this section finds the line (nbostart) where the nbo data is located
            for i in range(len(filecont)-1,0,-1): #search the file contents for the phrase "beta spin orbitals" to check for open shell molecules
                if re.search(nbo_os_pattern,filecont[i]) and skip == 0: 
                    skip = 2 # retrieve only combined orbitals NPA in open shell molecules 
                if npa_pattern.search(filecont[i]): #search the file content for the phrase which indicates the start of the NBO section 
                    if skip != 0:
                        skip = skip-1
                        continue
                    nbostart = i + 6 #skips the set number of lines between the search key and the start of the table
                    break      
            if nbostart == 0: 
                error = "****no Natural Population Analysis found in: " + str(row['log_name']) + ".log"
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'NBO_charge_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                nbo_dataframe = nbo_dataframe.append(row_i, ignore_index=True)
                continue
                
            #this section splits the table where nbo data is located into just the atom number and charge to generate a list of lists (nbo)
            ls = []
            for line in filecont[nbostart:]:
                if "==" in line: break
                ls = [str.split(line)[1],str.split(line)[2]] 
                nbo.append(ls)  
            
            #this uses the nbo list to return only the charges for only the atoms of interest as a list (nboout)
            nboout = get_specdata(atomnum_list,nbo)
            
            #this adds the data from the nboout into the new property df
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'NBO_charge_'+str(a_list[a]): nboout[a]}
                row_i.update(entry)
            nbo_dataframe = nbo_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire NBO charges for:', row['log_name'], ".log")
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'NBO_charge_'+str(a_list[a]): "no data"}
                row_i.update(entry)
            nbo_dataframe = nbo_dataframe.append(row_i, ignore_index=True)
    print("NBO function has completed for", a_list)
    return(pd.concat([dataframe, nbo_dataframe], axis = 1))
    
def get_nmr(dataframe, a_list): # a function to get the nbo for all atoms (a_list, form ["C1", "C4", "O2"]) in a dataframe that contains file name and atom number
    nmr_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        
        try: #try to get the data
            atom_list = [] 
            for new_a in a_list: 
                new_atom = row[str(new_a)] #the atom number (e.g., 16) to add to the list is the df entry of this row for the labeled atom (e.g., "C1")
                atom_list.append(str(new_atom)) #append that to atom_list to make a list of the form [16, 17, 29]
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'NMR_shift_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                nmr_dataframe = nmr_dataframe.append(row_i, ignore_index=True)
                continue
            
            #determining the locations/values for start and end of NMR section
            start,end,i = 0,0,0
            if nmrstart_pattern in filecont:
                start = filecont.index(nmrstart_pattern)+1
                for i in range(start,len(filecont),1):
                    if nmrend_pattern.search(filecont[i]) or nmrend_pattern_os.search(filecont[i]):
                        end = i
                        break
            if start == 0:
                error = "****no NMR data found in file: " + str(row['log_name']) + ".log"
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'NMR_shift_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                nmr_dataframe = nmr_dataframe.append(row_i, ignore_index=True)
                continue
                
            atoms = int((end - start)/5) #total number of atoms in molecule (there are 5 lines generated per atom)
            nmr = []
            for atom in range(atoms):
                element = str.split(filecont[start+5*atom])[1]
                shift_s = str.split(filecont[start+5*atom])[4]
                nmr.append([element,shift_s])
            #atom_list = ["1", "2", "3"]
            nmrout = get_specdata(atom_list,nmr) #revisit
            #print(nmrout)
            
            #this adds the data from the nboout into the new property df
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'NMR_shift_'+str(a_list[a]): nmrout[a]}
                row_i.update(entry)
            nmr_dataframe = nmr_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire NMR shifts for:', row['log_name'], ".log")
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'NMR_shift_'+str(a_list[a]): "no data"}
                row_i.update(entry)
            nmr_dataframe = nmr_dataframe.append(row_i, ignore_index=True)
    print("NMR function has completed for", a_list)
    return(pd.concat([dataframe, nmr_dataframe], axis = 1))
    
def get_angles(dataframe,angle_list): # a function to get the angles for all atoms (angle_list, form [[O3, C1, O2], [C4, C1, O3]]) in a dataframe that contains file name and atom number
    angle_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in
    
    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try:     
            #parsing the angle list from input line
            anglenums_list = [] 
            for angle in angle_list: 
                atomnum_list = [] #the atom numbers for an angle (e.g., 17 16 18) are collected from the df using the input list (e.g., ["O3", "C1", "O2"])
                for atom in angle:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                anglenums_list.append(atomnum_list) #append atomnum_list for each angle to make a list of the form [['17', '16', '18'], ['15', '16', '17']]
            
            angletitle_list = []
            for angle in angle_list:
                angletitle = str(angle[0]) + "_" + str(angle[1]) + "_" + str(angle[2])
                angletitle_list.append(angletitle)
            
            log_file = row['log_name'] #read file name from df
            streams, error = get_outstreams(log_file)
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(anglenums_list)):
                    entry = {'angle_'+str(angletitle_list[a]) + '(°)': "no data"}
                    row_i.update(entry)
                angle_dataframe = angle_dataframe.append(row_i, ignore_index=True)
                continue
            
            geom = get_geom(streams)
           
            #checks for if the wrong number of atoms are input, input is not of the correct form, or calls atom numbers that do not exist in the molecule.
            error = ""
            for angle in anglenums_list:
                if len(angle)%3 != 0:
                    error = "****Number of atom inputs given for angle is not divisible by three. " + str(len(angle)) + " atoms were given. "
                for atom in angle:
                    if not atom.isdigit():
                        error += "**** " + atom + ": Only numbers accepted as input for angles"
                    if int(atom) > len(geom):
                        error += "**** " + atom + " is out of range. Maximum valid atom number: " + str(len(geom)+1) + " "
                if error != "": print(error)
            
            anglesout = []
            for angle in anglenums_list:
                a = geom[int(angle[0])-1][:4] # atom coords
                b = geom[int(angle[1])-1][:4] 
                c = geom[int(angle[2])-1][:4]
                ba = np.array(a[1:]) - np.array(b[1:])
                bc = np.array(c[1:]) - np.array(b[1:])	  
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                anglevalue = np.arccos(cosine_angle)

                anglesout.append(float(round(np.degrees(anglevalue),3)))

            #this adds the data from the anglesout into the new property df
            row_i = {}
            for a in range(0, len(anglenums_list)):
                entry = {'angle_'+str(angletitle_list[a]) + '(°)': anglesout[a]}
                row_i.update(entry)
            angle_dataframe = angle_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire angles for:', row['log_name'], ".log")
            row_i = {}
            try:
                for a in range(0, len(anglenums_list)):
                    entry = {'angle_'+str(angletitle_list[a]) + '(°)': "no data"}
                    row_i.update(entry)
                angle_dataframe = angle_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("Angles function has completed for", angle_list)
    return(pd.concat([dataframe, angle_dataframe], axis = 1))
    
def get_dihedral(dataframe,dihedral_list): # a function to get the dihedrals for all atoms (dihederal_list, form [[O2, C1, O3, H5], [C4, C1, O3, H5]]) in a dataframe that contains file name and atom number
    dihedral_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in
    
    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try:
            #parsing the dihedral list from input line
            dihedralnums_list = [] 
            for dihedral in dihedral_list: 
                atomnum_list = [] #the atom numbers for a dihedral (e.g., 18 16 17 50) are collected from the df using the input list (e.g., ["O2", "C1", "O3", "H5"])
                for atom in dihedral:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                dihedralnums_list.append(atomnum_list) #append atomnum_list for each dihedral to make a list of the form [['18', '16', '17', '50'], ['18', '16', '17', '50']]
            dihedraltitle_list = []
            for dihedral in dihedral_list:
                dihedraltitle = str(dihedral[0]) + "_" + str(dihedral[1]) + "_" + str(dihedral[2]) + "_" +str(dihedral[3])
                dihedraltitle_list.append(dihedraltitle)
                
            log_file = row['log_name'] #read file name from df
            streams, error = get_outstreams(log_file)
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(dihedralnums_list)):
                    entry = {'dihedral_'+str(dihedraltitle_list[a]) + '(°)': "no data"}
                    row_i.update(entry)
                dihedral_dataframe = dihedral_dataframe.append(row_i, ignore_index=True)
                continue
            geom = get_geom(streams)
            
            #checks for if the wrong number of atoms are input, input is not of the correct form, or calls atom numbers that do not exist in the molecule.
            error = ""
            for dihedral in dihedralnums_list:
                if len(dihedral)%4 != 0:
                    error = "****Number of atom inputs given for dihedral angle is not divisible by four. " + str(len(dihedral)) + " atoms were given. "
                for atom in dihedral:
                    if not atom.isdigit():
                        error += "**** " + atom + ": Only numbers accepted as input for dihedral angles"
                    if int(atom) > len(geom):
                        error += "**** " + atom + " is out of range. Maximum valid atom number: " + str(len(geom)+1) + " "
                if error != "": print(error)
            
            dihedralsout = []
            for dihedral in dihedralnums_list:
                a = geom[int(dihedral[0])-1][:4] # atom coords
                b = geom[int(dihedral[1])-1][:4] 
                c = geom[int(dihedral[2])-1][:4]
                d = geom[int(dihedral[3])-1][:4]
                
                ab = np.array([a[1]-b[1],a[2]-b[2],a[3]-b[3]]) # vectors
                bc = np.array([b[1]-c[1],b[2]-c[2],b[3]-c[3]])
                cd = np.array([c[1]-d[1],c[2]-d[2],c[3]-d[3]])
                
                n1 = np.cross(ab,bc) # normal vectors
                n2 = np.cross(bc,cd)

                dihedral = round(np.degrees(np.arccos(np.dot(n1,n2) / (np.linalg.norm(n1)*np.linalg.norm(n2)))),3)
                dihedralsout.append(float(dihedral))
            
            #this adds the data from the dihedralsout into the new property df
            row_i = {}
            for a in range(0, len(dihedralnums_list)):
                entry = {'dihedral_'+str(dihedraltitle_list[a]) + '(°)': dihedralsout[a]}
                row_i.update(entry)
            dihedral_dataframe = dihedral_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire dihedral angles for:', row['log_name'], ".log")
            row_i = {}
            try:
                for a in range(0, len(dihedralnums_list)):
                    entry = {'dihedral_'+str(dihedraltitle_list[a]) + '(°)': "no data"}
                    row_i.update(entry)
                dihedral_dataframe = dihedral_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("Dihedral function has completed for", dihedral_list)
    return(pd.concat([dataframe, dihedral_dataframe], axis = 1))
    
def get_distance(dataframe,dist_list): # a function to get the distances for all atoms (dist_list, form [[C1, O2], [C4, C1]]) in a dataframe that contains file name and atom number
    dist_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in
    
    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try:
            #parsing the distances list from input line
            distnums_list = [] 
            for dist in dist_list: 
                atomnum_list = [] #the atom numbers for a distance (e.g., 18 16 16 15) are collected from the df using the input list (e.g., ["O2", "C1", "O3", "H5"])
                for atom in dist:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                distnums_list.append(atomnum_list) #append atomnum_list for each distance to make a list of the form [['18', '16'], ['16', '15']]
            
            disttitle_list = []
            for dist in dist_list:
                disttitle = str(dist[0]) + "_" + str(dist[1])
                disttitle_list.append(disttitle)
                
            log_file = row['log_name'] #read file name from df
            streams, error = get_outstreams(log_file)
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(distnums_list)):
                    entry = {'distance_' + str(disttitle_list[a]) + '(Å)': "no data"} 
                    row_i.update(entry)
                dist_dataframe = dist_dataframe.append(row_i, ignore_index=True)
                continue
            geom = get_geom(streams)
            
            
            #checks for if the wrong number of atoms are input, input is not of the correct form, or calls atom numbers that do not exist in the molecule.
            error = ""
            for dist in distnums_list:
                if len(dist)%2 != 0:
                    error = "****Number of atom inputs given for distance is not divisible by two. " + str(len(dist)) + " atoms were given. "
                for atom in dist:
                    if not atom.isdigit():
                        error += "**** " + atom + ": Only numbers accepted as input for distances"
                    if int(atom) > len(geom):
                        error += "**** " + atom + " is out of range. Maximum valid atom number: " + str(len(geom)+1) + " "
                if error != "": print(error)

            distout = []
            for dist in distnums_list:
                a = geom[int(dist[0])-1][:4] # atom coords
                b = geom[int(dist[1])-1][:4] 
                ba = np.array(a[1:]) - np.array(b[1:])
                dist = round(np.linalg.norm(ba),5)
                distout.append(float(dist))
                
            #this adds the data from the distout into the new property df
            row_i = {}
            for a in range(0, len(distnums_list)):
                entry = {'distance_' + str(disttitle_list[a]) + '(Å)': distout[a]}
                row_i.update(entry)
            dist_dataframe = dist_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire distance for:', row['log_name'], ".log")
            row_i = {}
            try:
                for a in range(0, len(distnums_list)):
                    entry = {'distance_' + str(disttitle_list[a]) + '(Å)': "no data"} 
                    row_i.update(entry)
                dist_dataframe = dist_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("Distance function has completed for", dist_list)
    return(pd.concat([dataframe, dist_dataframe], axis = 1))
    
def get_enthalpies(dataframe): # gets thermochemical data from freq jobs
    enthalpy_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in
    
    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try: #try to get the data
            log_file = row['log_name'] #read file name from df
            filecont = get_filecont(log_file) #read the contents of the log file
            
            evals = []
            error = "no thermochemical data found;;"
            e_hf,ezpe,h,g = 0,0,0,0
            for i in range(len(filecont)-1): #uses the zero_pattern that denotes this section to gather relevant energy terms
                if zero_pattern.search(filecont[i]):
                    e_hf = round(-eval(str.split(filecont[i-4])[-2]) + ezpe,6)
                    evals.append(e_hf)
                    ezpe = eval(str.split(filecont[i])[-1])
                    evals.append(ezpe)
                    h = eval(str.split(filecont[i+2])[-1])
                    evals.append(h)
                    g = eval(str.split(filecont[i+3])[-1])
                    evals.append(g)
                    error = ""

            #this adds the data from the energy_values list (evals) into the new property df
            row_i = {'ZP_correction(Hartree)': evals[0], 'E_ZPE(Hartree)': evals[1], 'H(Hartree)': evals[2], 'G(Hartree)': evals[3]}
            #print(row_i)
            
            enthalpy_dataframe = enthalpy_dataframe.append(row_i, ignore_index=True)
        except:
            print('Unable to acquire enthalpies for:', row['log_name'], ".log")
    print("Enthalpies function has completed")
    return(pd.concat([dataframe, enthalpy_dataframe], axis = 1))

def get_time(dataframe): # gets wall time and CPU for all jobs
    time_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in
    
    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try: #try to get the data
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {'CPU_time_total(hours)': "no data", 'Wall_time_total(hours)': "no data"}
                time_dataframe = time_dataframe.append(row_i, ignore_index=True)
                continue
                
            cputime,walltime = 0,0
            timeout = []
            for line in filecont:
                if cputime_pattern.search(line):
                    lsplt = str.split(line)
                    cputime = float(lsplt[-2])/3600 + float(lsplt[-4])/60 + float(lsplt[-6]) + float(lsplt[-8])*24
                    timeout.append(round(cputime,5))
                if walltime_pattern.search(line):
                    lsplt = str.split(line)
                    walltime = float(lsplt[-2])/3600 + float(lsplt[-4])/60 + float(lsplt[-6]) + float(lsplt[-8])*24
                    timeout.append(walltime)
            CPU_time = 0 
            Wall_time = 0
            for i in range(len(timeout)):
                if i%2 == 0: 
                    CPU_time += timeout[i]
                if i%2 != 0:
                    Wall_time += timeout[i]

            #this adds the data from the CPU_time and Wall_time into the property df
            row_i = {'CPU_time_total(hours)': CPU_time, 'Wall_time_total(hours)': Wall_time}
            time_dataframe = time_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire CPU time and wall time for:', row['log_name'], ".log")
            row_i = {'CPU_time_total(hours)': "no data", 'Wall_time_total(hours)': "no data"}
            time_dataframe = time_dataframe.append(row_i, ignore_index=True)
    print("Time function has completed")
    return(pd.concat([dataframe, time_dataframe], axis = 1))

def get_frontierorbs(dataframe): # homo,lumo energies and derived values of last job in file  
    frontierorbs_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in
    
    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try: #try to get the data
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {'HOMO': "no data", 'LUMO': "no data", "μ": "no data", "η": "no data", "ω": "no data"}
                frontierorbs_dataframe = frontierorbs_dataframe.append(row_i, ignore_index=True)
                continue
                
            frontierout = []
            index = 0
            for line in filecont[::-1]:
                if homo_pattern.search(line):
                    index += 1 #index ensures only the first entry is included
                    if index == 1: 
                        homo = float(str.split(line)[-1])
                        lumo = float(str.split(filecont[filecont.index(line)+1])[4])
                        mu = (homo+lumo)/2 # chemical potential or negative of molecular electronegativity
                        eta = lumo-homo # hardness/softness
                        omega = round(mu**2/(2*eta),5) # electrophilicity index
                        frontierout.append(homo)
                        frontierout.append(lumo)
                        frontierout.append(mu)
                        frontierout.append(eta)
                        frontierout.append(omega)
                    
            #this adds the data from the frontierout into the new property df
            row_i = {'HOMO': frontierout[0], 'LUMO': frontierout[1], "μ": frontierout[2], "η": frontierout[3], "ω": frontierout[4]}
            frontierorbs_dataframe = frontierorbs_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire frontier orbitals for:', row['log_name'], ".log")
            row_i = {'HOMO': "no data", 'LUMO': "no data", "μ": "no data", "η": "no data", "ω": "no data"}
            frontierorbs_dataframe = frontierorbs_dataframe.append(row_i, ignore_index=True)
    print("Frontier orbitals function has completed")
    return(pd.concat([dataframe, frontierorbs_dataframe], axis = 1))

def get_volume(dataframe): #gets the molar volume of the molecule
    volume_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in
    
    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try: #try to get the data
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {'volume(Bohr_radius³/mol)': "no data"}
                volume_dataframe = volume_dataframe.append(row_i, ignore_index=True)
                continue
        
            volume = []
            for line in filecont:   
                if volume_pattern.search(line):
                    volume.append(line.split()[3])
            #this adds the data into the new property df
            row_i = {'volume(Bohr_radius³/mol)': float(volume[0])}
            volume_dataframe = volume_dataframe.append(row_i, ignore_index=True)
            
        except:
            print('****Unable to acquire volume for:', row['log_name'], ".log")
            row_i = {'volume(Bohr_radius³/mol)': "no data"}
            volume_dataframe = volume_dataframe.append(row_i, ignore_index=True)
    print("Volume function has completed")
    return(pd.concat([dataframe, volume_dataframe], axis = 1))


def get_polarizability(dataframe): # polarizability isotropic and anisotropic 
    polarizability_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in
    
    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try: #try to get the data
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {'polar_iso(au)': "no data", 'polar_aniso(au)': "no data"}
                polarizability_dataframe = polarizability_dataframe.append(row_i, ignore_index=True)
                continue
        
            polarout = []
            for i in range(len(filecont)-1,1,-1):
                if polarizability_pattern.search(filecont[i]):
                    alpha_iso = float(filecont[i+4].split()[1].replace("D","E"))
                    alpha_aniso = float(filecont[i+5].split()[1].replace("D","E"))
                    polarout.append(alpha_iso)
                    polarout.append(alpha_aniso)
                                               
                                               
            #this adds the data from the polarout into the new property df
            row_i = {'polar_iso(au)': polarout[0], 'polar_aniso(au)': polarout[1]}
            polarizability_dataframe = polarizability_dataframe.append(row_i, ignore_index=True)
            
        except:
            print('****Unable to acquire polarizability for:', row['log_name'], ".log")
            row_i = {'polar_iso(au)': "no data", 'polar_aniso(au)': "no data"}
            polarizability_dataframe = polarizability_dataframe.append(row_i, ignore_index=True)
    print("Polarizability function has completed")
    return(pd.concat([dataframe, polarizability_dataframe], axis = 1))

def get_planeangle(dataframe,planeangle_list): # a function to get the plane angles for all atoms (dihederal_list, form [[O2, C1, O3, H5], [C4, C1, O3, H5]]) in a dataframe that contains file name and atom number
    planeangle_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in
    
    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try:
            #parsing the plane angle list from input line
            planeanglenums_list = [] 
            for planeangle in planeangle_list: 
                atomnum_list = [] #the atom numbers for a plane angle (e.g., 18 16 17 50) are collected from the df using the input list (e.g., ["O2", "C1", "O3", "H5"])
                for atom in planeangle:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                planeanglenums_list.append(atomnum_list) #append atomnum_list for each plane angle to make a list of the form [['18', '16', '17', '50'], ['18', '16', '17', '50']]
            
            planeangletitle_list = []
            for planeangle in planeangle_list:
                planeangletitle = str(planeangle[0]) + "_" + str(planeangle[1]) + "_" + str(planeangle[2]) + "_&_" +str(planeangle[3])+ "_" + str(planeangle[4]) + "_" +str(planeangle[5])
                planeangletitle_list.append(planeangletitle)
            
            log_file = row['log_name'] #read file name from df
            streams, error = get_outstreams(log_file)
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(planeanglenums_list)):
                    entry = {'planeangle_'+str(planeangletitle_list[a]) + '(°)': "no data"}
                    row_i.update(entry)
                planeangle_dataframe = planeangle_dataframe.append(row_i, ignore_index=True)
                continue
                
            geom = get_geom(streams)

            #checks for if the wrong number of atoms are input, input is not of the correct form, or calls atom numbers that do not exist in the molecule.
            error = ""
            for planeangle in planeanglenums_list:
                if len(planeangle)%6 != 0:
                    error = "****Number of atom inputs given for plane angle is not divisible by six. " + str(len(planeangle)) + " atoms were given. "
                for atom in planeangle:
                    if not atom.isdigit():
                        error += "**** " + atom + ": Only numbers accepted as input for plane angles"
                    if int(atom) > len(geom):
                        error += "**** " + atom + " is out of range. Maximum valid atom number: " + str(len(geom)+1) + " "
                if error != "": print(error)

            planeanglesout = []
            for planeangle in planeanglenums_list:
                a = geom[int(planeangle[0])-1][:4] 
                b = geom[int(planeangle[1])-1][:4] 
                c = geom[int(planeangle[2])-1][:4] 
                d = geom[int(planeangle[3])-1][:4] 
                e = geom[int(planeangle[4])-1][:4] 
                f = geom[int(planeangle[5])-1][:4] 

                ab = np.array([a[1]-b[1],a[2]-b[2],a[3]-b[3]]) # vectors
                bc = np.array([b[1]-c[1],b[2]-c[2],b[3]-c[3]])
                de = np.array([d[1]-e[1],d[2]-e[2],d[3]-e[3]])
                ef = np.array([e[1]-f[1],e[2]-f[2],e[3]-f[3]])

                n1 = np.cross(ab,bc) # Normal vectors
                n2 = np.cross(de,ef)

                planeangle_value = round(np.degrees(np.arccos(np.dot(n1,n2) / (np.linalg.norm(n1)*np.linalg.norm(n2)))),3)
                planeangle_value = min(abs(planeangle_value),abs(180-planeangle_value))
                planeanglesout.append(planeangle_value)
                
            #this adds the data from the planeanglesout into the new property df
            row_i = {}
            for a in range(0, len(planeanglenums_list)):
                entry = {'planeangle_'+str(planeangletitle_list[a]) + '(°)': planeanglesout[a]}
                row_i.update(entry)
            planeangle_dataframe = planeangle_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire plane angle for:', row['log_name'], ".log")
            row_i = {}
            try:
                for a in range(0, len(planeanglenums_list)):
                    entry = {'planeangle_'+str(planeangletitle_list[a]) + '(°)': "no data"}
                    row_i.update(entry)
                planeangle_dataframe = planeangle_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("Plane angle function has completed for", planeangle_list)
    return(pd.concat([dataframe, planeangle_dataframe], axis = 1))
    
def get_dipole(dataframe):
    dipole_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in
    
    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try: #try to get the data
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {'dipole(Debye)': "no data"}
                dipole_dataframe = dipole_dataframe.append(row_i, ignore_index=True)
                continue
                
            dipole = []
            for i in range(len(filecont)-1,0,-1): #search filecont in backwards direction
                if dipole_pattern in filecont[i]:
                    dipole.append(float(str.split(filecont[i+1])[-1]))
            #this adds the data from the first dipole entry (corresponding to the last job in the file) into the new property df
            row_i = {'dipole(Debye)': dipole[0]}
            dipole_dataframe = dipole_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire dipole for:', row['log_name'], ".log")
            row_i = {'dipole(Debye)': "no data"}
            dipole_dataframe = dipole_dataframe.append(row_i, ignore_index=True)
    print("Dipole function has completed")
    return(pd.concat([dataframe, dipole_dataframe], axis = 1))
    
def get_SASA(dataframe): #uses morfeus to calculate solvent accessible surface area in a dataframe that contains file name
    #if you want to SASA with different probe radii, morfeus has this functionality, but it has not been implemented here
    sasa_dataframe = pd.DataFrame(columns=[])
    
    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']
            streams, error = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
            if error != "":
                print(error)
                row_i = {'SASA_surface_area(Å²)': "no data", 
                     'SASA_volume(Å³)': "no data",
                     'SASA_sphericity': "no data"} 
                sasa_dataframe = sasa_dataframe.append(row_i, ignore_index=True) 
                continue
                
            log_coordinates = get_geom(streams)
            elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
            coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])

            sasa = SASA(elements,coordinates) #calls morfeus
            
            sphericity = np.cbrt((36*math.pi*sasa.volume**2))/sasa.area
            
            row_i = {'SASA_surface_area(Å²)': sasa.area, 
                     'SASA_volume(Å³)': sasa.volume, #volume inside the solvent accessible surface area
                     'SASA_sphericity': sphericity} 
            sasa_dataframe = sasa_dataframe.append(row_i, ignore_index=True)   
        except:
            print('****Unable to acquire SASA parameters for:', row['log_name'], ".log")
            row_i = {'SASA_surface_area(Å²)': "no data", 
                     'SASA_volume(Å³)': "no data",
                     'SASA_sphericity': "no data"} 
            sasa_dataframe = sasa_dataframe.append(row_i, ignore_index=True) 
    print("SASA function has completed")
    return(pd.concat([dataframe, sasa_dataframe], axis = 1))
    
def get_goodvibes_e(dataframe, temp): #uses goodvibes to calculate thermochemical energy values, requires frequency job
    e_dataframe = pd.DataFrame(columns=[])
    options = gv.GVOptions()
    options.spc = 'link' 
    options.temperature = temp
    
    # create a text file for all output (required)
    log = io.Logger("Goodvibes", 'output', False)
    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']
            file_data = io.getoutData(str(log_file) + ".log", options)
            # Carry out the thermochemical analysis - auto-detect the vibrational scaling factor
            options.freq_scale_factor = False # turns of default value of 1
            level_of_theory = [file_data.functional + '/' + file_data.basis_set]
            options.freq_scale_factor, options.mm_freq_scale_factor = gv.get_vib_scale_factor(level_of_theory, options, log)
            bbe_val = thermo.calc_bbe(file_data, options)
            properties = ['sp_energy', 'zpe', 'enthalpy', 'entropy', 'qh_entropy', 'gibbs_free_energy', 'qh_gibbs_free_energy']
            vals = [getattr(bbe_val, k) for k in properties]
            
            row_i = {'E_spc (Hartree)': vals[0],
                    'ZPE(Hartree)': vals[1],
                    'H_spc(Hartree)': vals[2],
                    'T*S': vals[3]*options.temperature, 
                    'T*qh_S': vals[4]*options.temperature, 
                    'G(T)_spc(Hartree)': vals[5], 
                    'qh_G(T)_spc(Hartree)': vals[6],
                    'T': options.temperature}
            
            e_dataframe = e_dataframe.append(row_i, ignore_index=True)   
        except:
            print("")    
            print('****Unable to acquire goodvibes energies for:', row['log_name'], ".log")
            row_i = {'E_spc (Hartree)': "no data",
                    'ZPE(Hartree)': "no data",
                    'H_spc(Hartree)': "no data",
                    'T*S': "no data", 
                    'T*qh_S': "no data", 
                    'G(T)_spc(Hartree)': "no data", 
                    'qh_G(T)_spc(Hartree)': "no data",
                    'T': "no data"}
            e_dataframe = e_dataframe.append(row_i, ignore_index=True)  
    print("")
    print("Goodvibes function has completed")
    return(pd.concat([dataframe, e_dataframe], axis = 1))
    
class IR:
    def __init__(self,filecont,start,col,len):
        self.freqno = int(filecont[start].split()[-3+col])
        self.freq = float(filecont[start+2].split()[-3+col])
        self.int = float(filecont[start+5].split()[-3+col])
        self.deltas = []
        atomnos = []
        for a in range(len-7):
            atomnos.append(filecont[start+7+a].split()[1])
            x = float(filecont[start+7+a].split()[3*col+2])
            y = float(filecont[start+7+a].split()[3*col+3])
            z = float(filecont[start+7+a].split()[3*col+4])
            self.deltas.append(np.linalg.norm([x,y,z]))
            

def get_IR(dataframe, a1, a2, freqmin, freqmax, intmin, intmax, threshold): # a function to get IR values for a pair of atoms at a certain freq and intensity
    IR_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in
    pair_label = str(a1)+"_"+str(a2)
    
    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try:
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file)
            if error != "":
                print(error)
                row_i = {'IR_freq_'+str(pair_label): "no data"}
                IR_dataframe = IR_dataframe.append(row_i, ignore_index=True)
                continue
            #this changes a1 and a2 (of the form "C1" and "O3") to atomnum_pair (of the form [17, 18])
            atom1 = row[str(a1)]
            atom2 = row[str(a2)]
            
            #this section finds where all IR frequencies are located in the log file
            frq_len = 0
            frq_end = 0
            for i in range(len(filecont)):
                if frqs_pattern.search(filecont[i]) and frq_len == 1: #subsequent times it finds the pattern, it recognizes the frq_len
                    frq_len = i -3 - frq_start
                if frqs_pattern.search(filecont[i]) and frq_len == 0: #first time it finds the pattern it will set frq_start
                    frq_start = i-3
                    frq_len = 1
                if frqsend_pattern.search(filecont[i]): #finds the end pattern
                    frq_end = i-3
 
            nfrq = filecont[frq_end-frq_len+1].split()[-1]
            blocks = int((frq_end + 1 - frq_start)/frq_len) 
            irdata = []   # list of objects IR contains: IR.freq, IR.int, IR.deltas = []
            
            for i in range(0, blocks):
                for j in range(len(filecont[i*frq_len+frq_start].split())):
                    irdata.append(IR(filecont,i*frq_len+frq_start,j,frq_len))
                
            irout = []
            for i in range(len(irdata)):
                if irdata[i].freq < freqmax and irdata[i].freq > freqmin and irdata[i].int > intmin and irdata[i].int < intmax and irdata[i].deltas[int(atom1)] >= threshold and irdata[i].deltas[int(atom2)] >= threshold:
                        irout = [irdata[i].freq, irdata[i].int]
                        
            #this adds the frequency data from the irout into the new property df
            row_i = {'IR_freq_'+str(pair_label): irout[0]}
            IR_dataframe = IR_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire IR frequencies for:', row['log_name'], ".log")
            row_i = {'IR_freq_'+str(pair_label): "no data"}
            IR_dataframe = IR_dataframe.append(row_i, ignore_index=True)
    print("IR function has completed for", a1, "and", a2)
    return(pd.concat([dataframe, IR_dataframe], axis = 1))
    
def get_buried_sterimol(dataframe, sterimol_list, r_buried): #uses morfeus to calculate sterimol L, B1, B5 for two input atoms for every entry in df
    sterimol_dataframe = pd.DataFrame(columns=[])
    r_buried -= 0.5 #the function adds 
    
    for index, row in dataframe.iterrows():
        try:
            #parsing the Sterimol axis defined in the list from input line
            sterimolnums_list = [] 
            for sterimol in sterimol_list: 
                atomnum_list = [] #the atom numbers use to collect sterimol values (e.g., [18 16 17 15]) are collected from the df using the input list (e.g., [["O2", "C1"], ["O3", "H5"]])
                for atom in sterimol:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                sterimolnums_list.append(atomnum_list) #append atomnum_list for each sterimol axis defined in the input to make a list of the form [['18', '16'], ['16', '15']]
            
            #this makes column headers based on Sterimol axis defined in the input line
            sterimoltitle_list = []
            for sterimol in sterimol_list:
                sterimoltitle = str(sterimol[0]) + "_" + str(sterimol[1])
                sterimoltitle_list.append(sterimoltitle)
                
            log_file = row['log_name']
            streams, error = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(sterimolnums_list)):
                    entry = {'Buried_Sterimol_L_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': "no data", 
                    'Buried_Sterimol_B1_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': "no data", 
                    'Buried_Sterimol_B5_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': "no data"}
                    row_i.update(entry)
                sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
                continue
                
            geom = get_geom(streams)
            
            #checks for if the wrong number of atoms are input, input is not of the correct form, or calls atom numbers that do not exist in the molecule
            error = ""
            for sterimol in sterimolnums_list:
                if len(sterimol)%2 != 0:
                    error = "Number of atom inputs given for Sterimol is not divisible by two. " + str(len(atoms)) + " atoms were given. "
                for atom in sterimol:
                    if not atom.isdigit():
                        error += " " + atom + ": Only numbers accepted as input for Sterimol"
                    if int(atom) > len(geom):
                        error += " " + atom + " is out of range. Maximum valid atom number: " + str(len(geom)+1) + " "
                if error != "": print(error)
                    
            elements = np.array([geom[i][0] for i in range(len(geom))])
            coordinates = np.array([np.array(geom[i][1:]) for i in range(len(geom))])
            
            #this collects Sterimol values for each pair of inputs
            sterimolout = []
            for sterimol in sterimolnums_list:
                sterimol_values = Sterimol(elements, coordinates, int(sterimol[0]), int(sterimol[1])) #calls morfeus
                sterimol_values.bury(method="delete", sphere_radius=float(r_buried))
                sterimolout.append(sterimol_values)
            
            #this adds the data from sterimolout into the new property df
            row_i = {}
            for a in range(0, len(sterimolnums_list)):
                entry = {'Buried_Sterimol_L_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': sterimolout[a].L_value, 
                'Buried_Sterimol_B1_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': sterimolout[a].B_1_value, 
                'Buried_Sterimol_B5_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': sterimolout[a].B_5_value}
                row_i.update(entry)
            sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire Morfeus Buried Sterimol parameters for:', row['log_name'], ".log")
            row_i = {}
            try:
                for a in range(0, len(sterimolnums_list)):
                    entry = {'Buried_Sterimol_L_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': "no data", 
                    'Buried_Sterimol_B1_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': "no data", 
                    'Buried_Sterimol_B5_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': "no data"}
                    row_i.update(entry)
                sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("Morfeus Buried Sterimol function has completed for", sterimol_list)
    return(pd.concat([dataframe, sterimol_dataframe], axis = 1))
    
def get_chelpg(dataframe, a_list): #a function to get the ChelpG ESP charges for all atoms (a_list, form ["C1", "C4", "O2"]) in a dataframe that contains file name and atom number
    chelpg_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try:#try to get the data
            atomnum_list = [] 
            for atom in a_list: 
                atomnum = row[str(atom)] #the atom number (e.g., 16) to add to the list is the df entry of this row for the labeled atom (e.g., "C1")
                atomnum_list.append(str(atomnum)) #append that to atomnum_list to make a list of the form [16, 17, 29]
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'ChelpG_charge_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                chelpg_dataframe = chelpg_dataframe.append(row_i, ignore_index=True)
                continue
                
            chelpgstart,chelpg,error,chelpgout = 0,False,"",[]
            
            #this section finds the line (chelpgstart) where the ChelpG data is located
            for i in range(len(filecont)-1,0,-1):
                if chelpg2_pattern.search(filecont[i]):
                    chelpgstart = i
                if chelpg1_pattern.search(filecont[i]):
                    chelpg = True
                    break
            if chelpgstart != 0 and chelpg == False:
                error = "****Other ESP scheme than ChelpG used in: " + str(log_file) + ".log"
            if chelpgstart == 0: 
                error = "****no ChelpG ESP charge analysis found in: "+ str(log_file) + ".log"
            if error != "":    
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'ChelpG_charge_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                chelpg_dataframe = chelpg_dataframe.append(row_i, ignore_index=True)
                continue
                
            for atom in atomnum_list:
                if atom.isnumeric():
                    chelpgout.append(filecont[chelpgstart+int(atom)+2].split()[-1])
            
            #this adds the data from the chelpgout into the new property df
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'ChelpG_charge_'+str(a_list[a]): chelpgout[a]}
                row_i.update(entry)
            chelpg_dataframe = chelpg_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire ChelpG charges for:', row['log_name'], ".log")
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'ChelpG_charge_'+str(a_list[a]): "no data"}
                row_i.update(entry)
            chelpg_dataframe = chelpg_dataframe.append(row_i, ignore_index=True)
    print("ChelpG function has completed for", a_list)
    return(pd.concat([dataframe, chelpg_dataframe], axis = 1))
    
def get_hirshfeld(dataframe,a_list): #a function to get the Hirshfeld charge, CM5 charge, and atomic dipole for all atoms (a_list, form ["C1", "C4", "O2"]) in a dataframe that contains file name and atom number
    hirsh_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try:#try to get the data
            atomnum_list = [] 
            for atom in a_list: 
                atomnum = row[str(atom)] #the atom number (e.g., 16) to add to the list is the df entry of this row for the labeled atom (e.g., "C1")
                atomnum_list.append(str(atomnum)) #append that to atomnum_list to make a list of the form [16, 17, 29]
            
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'Hirsh_charge_'+str(a_list[a]): "no data",
                            'Hirsh_CM5_charge_'+str(a_list[a]): "no data",
                            'Hirsh_atom_dipole_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                hirsh_dataframe = hirsh_dataframe.append(row_i, ignore_index=True)
                continue
                
            hirshstart,error,hirshout = 0,False,[]
            
            #this section finds the line (chelpgstart) where the ChelpG data is located
            for i in range(len(filecont)-1,0,-1):
                if hirshfeld_pattern.search(filecont[i]):
                    hirshstart = i
                    break
            if hirshstart == 0: 
                error = "****no Hirshfeld Population Analysis found in: " + str(log_file) + ".log"
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'Hirsh_charge_'+str(a_list[a]): "no data",
                        'Hirsh_CM5_charge_'+str(a_list[a]): "no data",
                        'Hirsh_atom_dipole_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                hirsh_dataframe = hirsh_dataframe.append(row_i, ignore_index=True)
                continue
    
            for atom in atomnum_list:
                if atom.isnumeric():
                    cont = filecont[hirshstart+int(atom)+1].split()
                    qh = cont[2] #using 0-indexing, this gets the value for Hirshfeld charge from the 2nd column
                    qcm5 = cont[7] #using 0-indexing, this gets the value for CM5 charge from the 7th column
                    d = np.linalg.norm(np.array((cont[4:8])))
                    hirshout.append([str(qh),str(qcm5),str(d)])

            #this adds the data from the hirshout into the new property df
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'Hirsh_charge_'+str(a_list[a]): hirshout[a][0],
                        'Hirsh_CM5_charge_'+str(a_list[a]): hirshout[a][1],
                        'Hirsh_atom_dipole_'+str(a_list[a]): hirshout[a][2]}
                row_i.update(entry)
            hirsh_dataframe = hirsh_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire Hirshfeld properties for:', row['log_name'], ".log")
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'Hirsh_charge_'+str(a_list[a]): "no data",
                        'Hirsh_CM5_charge_'+str(a_list[a]): "no data",
                        'Hirsh_atom_dipole_'+str(a_list[a]): "no data"}
                row_i.update(entry)
            hirsh_dataframe = hirsh_dataframe.append(row_i, ignore_index=True)
    print("Hirshfeld function has completed for", a_list)
    return(pd.concat([dataframe, hirsh_dataframe], axis = 1))       
    