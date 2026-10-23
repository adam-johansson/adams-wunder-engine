@echo off

if "%1"=="" echo You must provide an engine type on the command line (basesr, ddtflr etc)
if "%1"=="" goto end

del performanceWeight.txt 
del PerformanceResults.txt
del weicoErrorReport.txt

if %1==basesr (
  echo on
  copy input_files\dimensionsWeight_basesr.txt dimensionsWeight.txt
  copy input_files\performanceWeight_basesr.txt performanceWeight.txt
  del PlantCostResults.fo
    @echo off
  goto end
)

if %1==baselr (
  echo on
  copy input_files\dimensionsWeight_baselr.txt dimensionsWeight.txt
  copy input_files\performanceWeight_baselr.txt performanceWeight.txt
  del PlantCostResults.fo
    @echo off
  goto end
)

if %1==intercontinental (
  echo on
  copy input_files\dimensionsWeight_intcont.txt dimensionsWeight.txt
  copy input_files\performanceResults_intcont.txt performanceResults.txt
  del PlantCostResults.fo
    @echo off
  goto end
)

if %1==intraeuropean (
  echo on
  echo retrievingInterEuropean
  copy input_files\dimensionsWeight_inteur.txt dimensionsWeight.txt
  copy input_files\performanceResults_inteur.txt performanceResults.txt
  del PlantCostResults.fo
    @echo off
  goto end
)


rem if you got here the argument was not recognised
echo Command line argument not recognised


:end

