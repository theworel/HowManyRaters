outputdir=$1
infile=$2
morphoditaCommand="../morphodita-1.11.3/bin/"

${morphoditaCommand}run_morpho_analyze morphology_model use_guesser $infile | ./python/breakTag | ${morphoditaCommand}run_morpho_generate morphology_model use_guesser



run_morpho_generate morphology_model use_guesser 