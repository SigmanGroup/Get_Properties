# Get_Properties
Workflow scripts for automated collection of molecular properties, as well as atom- and bond-level properties for a conserved moiety of interest from Gaussian jobs. Post-processing allows for the collection of condensed descriptors for conformational ensembles: 
	* minimum property value
	* maximum property value
	* property value from the lowest energy conformer
	* Boltzmann-weighted average
	* Boltzmann-weighted standard deviation
	* range
	* property value from a specific conformer (e.g., Vbur(min) conformer)
Note, the user can specify an energy cutoff (in kcal/mol) to remove high-energy DFT-optimized conformers before post-processing.

## Conda Environment Setup
Uses Python/3.8.8
### One time setup instructions for Windows, Mac, and Linux users: 
    Step 1: Create a conda environment using the correct .yml file for your operating system: 
      Windows: conda env create -f gpenv_win.yml
      Mac: conda env create -f gpenv_mac.yml
	  Linux: conda env create -f gpenv_linux.yml
    Step 2: Activate it: 
      Windows: conda activate gpenv_win 
      Mac: conda activate gpenv_mac
	  Linux: conda activate gpenv_linux
    Step 3: Install GoodVibes (Jupyter Notebook branch) 
      git clone https://github.com/patonlab/goodvibes
      cd goodvibes
      git checkout GV2021
      python setup.py install 

Make sure the kernel is set to the correct environment when using the Jupyter Notebook.
    
## Properties
  * energies (GoodVibes)
  * nbo 
  * nmr 
  * angle
  * dihedral angle
  * distance
  * plane angle
  * total time
  * frontier molecular orbitals
  * volume
  * polarizability
  * dipole
  * Sterimol (ᴍᴏʀғᴇᴜs & DBSTEP)
  * buried Sterimol (ᴍᴏʀғᴇᴜs)
  * buried volume (ᴍᴏʀғᴇᴜs)
  * buried volume scan (ᴍᴏʀғᴇᴜs)
  * pyramidalization (ᴍᴏʀғᴇᴜs)
  * solvent accessible surface area (SASA) & sphericity (ᴍᴏʀғᴇᴜs)
  * Sterimol2Vec (DBSTEP)
  * Hirshfeld charges 
  * ChelpG
  * IR stretching frequency – *works for one stretch in the input range*

Be sure to include correct Gaussian input keywords for properties you will want (i.e., nbo, nmr, volume, etc.)

## Important Notes
  * This script assumes that you have conformational ensembles for multiple compounds and that your naming scheme has some prefix, compound identifier (i.e., number, letter, name), suffix, and conformer number (i.e., Ac1_1.log. Ac1_2.log. Ac2_1.log. Ac2_2.log).
    * If this is not the case, you can use a bulk renaming utility to adopt suitable names. 
  * This script is only intended to get properties for **linked** jobs.
    * If you don’t have linked jobs, get your own energies, read them in as an extra column in the atom map Excel or add them to the All_Conformer_Properties_example.xlsx.
      * Cannot get Gibbs energy for a single point job with solvent model.
	  * Cannot get thermodynamic properties without a frequency job.

## Acknowledgements
This work is developed by Brittany C. Haas and Melissa A. Hardy. Portions are adapted from code by David B. Vogt and Jordan P. Liles.

## Citing Get Properties
Please reference the DOI of our Zenodo repository with:
XXX

## License
Get Properties is freely available for both academic and commercial use under the MIT license.
