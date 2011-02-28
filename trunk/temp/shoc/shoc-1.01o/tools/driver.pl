#!/usr/bin/perl
use strict;

# Simple Script To Run SHOC Benchmarks
print "--- Welcome To The SHOC Benchmark Suite --- \n";

# Parse Command Line Arguments
my $numNodes    = getArg("-n");
my $devsPerNode = getArg("-d");
my $useCuda     = getStringArg("-cuda");
my $useOCL      = getStringArg("-opencl");
my $sizeClass   = getArg("-s");
my $help        = getStringArg("-help") ||
                  getStringArg("--help");
if ($help)
{
  usage();
}

# tell driver location of SHOC bin directory, allowing it to be called from 
# anywhere
my $bindir      = getArg("-bin-dir");

# don't execute tests, just read existing log files (for debugging driver)
my $readonly    = getStringArg("-read-only");

die "Please specify a size class between 1 and 4 (e.g. -s 1)"
  unless ( $sizeClass < 5 && $sizeClass > 0 );
die "Please specify -cuda or -opencl\n"
  unless ( ( $useCuda || $useOCL ) && !( $useCuda && $useOCL ) );

# Create a directory to save logs for each benchmark
if ( -d "./Logs" ) {
    my $tmp = system"echo \"Version 1.01\">Logs/version.txt ";
}
else {
   my $retval = system("mkdir Logs");
   die "Unable to create logs directory" unless $retval == 0;
}

# Find out the hostname
my $host_name = `hostname`;
chomp($host_name);
print "Hostname: $host_name \n";

# Print info about available devices
my $availDevices = printDevInfo($useCuda);
if ( $availDevices < 1 ) {
   print "Could not get number of devices; assuming $devsPerNode.\n";
   $availDevices = $devsPerNode;
}

print "\n--- Starting Benchmarks ---\n";

if ( $numNodes * $devsPerNode > 1 ) {
   parallelBM( $availDevices, $numNodes, $devsPerNode, $useCuda );
}
else {
   serialBM( $availDevices, $useCuda );
}

print "--- Finished --- \n";
print "Thank you for using SHOC.  If you encountered any ";
print "benchmark errors, please send a tarball of your Logs folder to";
print " shoc-help\@email.ornl.gov \n";

# Subroutine parallelBM(numNodes, devsPerNode)
# Purpose: Runs Parallel Benchmarks
# numNodes: number of nodes to use
# devsPerNode: the number of devices per node to use
sub parallelBM {

   my $availDevices = $_[0];
   my $numNodes     = $_[1];
   my $devsPerNode  = $_[2];
   my $useCuda      = $_[3];
   my $totalRanks   = $numNodes * $devsPerNode;

   my $numPasses = 1;
   if ( $devsPerNode != $availDevices )
   {
      $numPasses = $availDevices;
   }

   my $deviceArg = "";
   open( OUTFILE, ">results.csv" ) or die $!;
   for ( my $pass = 0 ; $pass < $numPasses ; $pass++ ) {
      if ( $numPasses == 1 ) {
         print "-- Single benchmark pass for all devices --\n";
         print OUTFILE "Using all devices for test\n";
      }
      else {
         print "-- Starting benchmark pass for device $pass --\n";
         print OUTFILE "\nUsing device $pass\n";
         $deviceArg = "-d $pass";
      }

      # Run all the level 0 benchmarks, and store their output in log files
      print "-- Running Level 0 Benchmarks --\n";

      # Run L0 Benchmarks
      print " * BusSpeedDownload\n";
      system(
         buildParCommand(
            "BusSpeedDownload", $totalRanks, $devsPerNode, $deviceArg
         )
      );
      print " * BusSpeedReadback\n";
      system(
         buildParCommand(
            "BusSpeedReadback", $totalRanks, $devsPerNode, $deviceArg
         )
      );
      print " * MaxFlops\n";
      system(
         buildParCommand(
            "MaxFlops", $totalRanks, $devsPerNode, $deviceArg
         )
      );
      print " * DeviceMemory\n";
      system(
         buildParCommand(
            "DeviceMemory", $totalRanks, $devsPerNode, $deviceArg
         )
      );

      # Kernel Compilation / Queue Delay not applicable for CUDA
      if ( !$useCuda ) {
          print " * KernelCompile\n";
         system(
            buildParCommand( "KernelCompile", $totalRanks, $devsPerNode ) );
          print " * QueueDelay\n";
         system(
            buildParCommand( "QueueDelay", $totalRanks, $devsPerNode ) );
      }

      # Run L1 Benchmarks
      print "-- Running Level 1 Benchmarks --\n";
      print " * FFT\n";
      system(
         buildParCommand( "FFT", $totalRanks, $devsPerNode, $deviceArg ) );
      print " * MD\n";
      system(
         buildParCommand( "MD", $totalRanks, $devsPerNode, $deviceArg ) );
      print " * Reduction\n";
      system(
         buildParCommand(
            "Reduction", $totalRanks, $devsPerNode, $deviceArg
         )
      );
      print " * S3D\n";
      system(
         buildParCommand( "S3D", $totalRanks, $devsPerNode, $deviceArg ) );
      print " * Scan\n";
      system(
         buildParCommand( "Scan", $totalRanks, $devsPerNode, $deviceArg ) );
      print " * SGEMM\n";
      system(
         buildParCommand( "SGEMM", $totalRanks, $devsPerNode, $deviceArg ) );
      print " * Sort\n";
      system(
         buildParCommand( "Sort", $totalRanks, $devsPerNode, $deviceArg ) );
      print " * Spmv\n";
      system(
         buildParCommand( "Spmv", $totalRanks, $devsPerNode, $deviceArg ) );
      print " * Triad\n";
      system(
         buildParCommand( "Triad", $totalRanks, $devsPerNode, $deviceArg ) );

      # Print some results
      print "-- Results --\n";
      print "- Level 0: \"Feeds and Speeds\" -\n";

      # Grep the log files for results
      my $avgFlops;
      if ($useCuda) {
          $avgFlops = findanymean( buildFileName( "MaxFlops", 0 ), "-SP" );
          #$avgDPFlops = findanymean( buildFileName( "MaxFlops", $i ), "-DP" );
      }
      else {
          $avgFlops = findanymean( buildFileName( "MaxFlops", 0 ), "-SP" );
          #$avgDPFlops = findanymean( buildFileName( "MaxFlops", $i ), "-DP" );
      }
      my $avgBandwidth = findmean(
         buildFileName( "DeviceMemory", 0 ),
         "readGlobalMemoryCoalesced(max)"
      );
      my $avgBusD = findmean( buildFileName( "BusSpeedDownload", 0 ),
         "DownloadSpeed(max)" );
      my $avgBusR = findmean( buildFileName( "BusSpeedReadback", 0 ),
         "ReadbackSpeed(max)" );
      my $avgKernel;
      my $avgDelay;
      if ( !$useCuda ) {
         $avgKernel = findmean( buildFileName( "KernelCompile", 0 ),
            "BuildProgram(min)" );
         $avgDelay =
           findmean( buildFileName( "QueueDelay", 0 ), "SSDelay(min)" );
      }
      my $avgSpFFT = findmean( buildFileName( "FFT", 0 ), "SP-FFT(max)" );
      my $avgDpFFT = findmean( buildFileName( "FFT", 0 ), "DP-FFT(max)" );
      my $avgMD    = findmean( buildFileName( "MD",  0 ), "MD-LJ(max)" );
      my $avgMD_DP = findmean( buildFileName( "MD",  0 ), "MD-LJ-DP(max)" );
      
      my $avgReduction =
         findmean( buildFileName( "Reduction", 0 ), "Reduction(max)" );
      my $avgReduction_DP =
         findmean( buildFileName( "Reduction", 0 ), "Reduction-DP(max)" );
      my $avgS3D = findmean( buildFileName( "S3D", 0 ), "S3D-SP(max)" );
      my $avgS3D_DP = findmean( buildFileName( "S3D", 0 ), "S3D-DP(max)" );
      my $avgScan = findmean( buildFileName( "Scan", 0 ), "Scan(max)" );
      my $avgScan_DP = findmean( buildFileName( "Scan", 0 ), "Scan-DP(max)" );
      my $avgSort = findmean( buildFileName( "Sort", 0 ), "Sort-Rate(max)" );
    
      my $avgCSRScalar = 
         findmean( buildFileName( "Spmv", 0 ), "CSR-Scalar-SP(max)" );
      my $avgCSRVector = 
         findmean( buildFileName( "Spmv", 0 ), "CSR-Vector-SP(max)" );
      my $avgEllpack   = 
         findmean( buildFileName( "Spmv", 0 ), "ELLPACKR-SP(max)" );
      my $avgCSRScalar_DP = 
         findmean( buildFileName( "Spmv", 0 ), "CSR-Scalar-DP(max)" );
      my $avgCSRVector_DP = 
         findmean( buildFileName( "Spmv", 0 ), "CSR-Vector-DP(max)" );
      my $avgEllpack_DP   = 
         findmean( buildFileName( "Spmv", 0 ), "ELLPACKR-DP(max)" );
    
      my $avgTriad =
        findmean( buildFileName( "Triad", 0 ), "TriadBdwth(max)" );
      my $avgSGEMM = findmean( buildFileName( "SGEMM", 0 ), "SGEMM-N(max)" );
      my $avgDGEMM = findmean( buildFileName( "SGEMM", 0 ), "DGEMM-N(max)" );

      # Do Some Basic Error Checking
      $avgFlops     = checkError($avgFlops);
      $avgBandwidth = checkError($avgBandwidth);
      $avgBusD      = checkError($avgBusD);
      $avgBusR      = checkError($avgBusR);
      if ( !$useCuda ) {
         $avgKernel = checkError($avgKernel);
         $avgDelay  = checkError($avgDelay);
      }
      
      $avgSpFFT        = checkError($avgSpFFT);
      $avgDpFFT        = checkError($avgDpFFT);
      $avgMD           = checkError($avgMD);
      $avgMD_DP        = checkError($avgMD_DP);
      $avgReduction    = checkError($avgReduction);
      $avgReduction_DP = checkError($avgReduction_DP);
      $avgS3D          = checkError($avgS3D);
      $avgS3D_DP       = checkError($avgS3D_DP);
      $avgScan         = checkError($avgScan);
      $avgScan_DP      = checkError($avgScan_DP);
      $avgSGEMM        = checkError($avgSGEMM);
      $avgDGEMM        = checkError($avgDGEMM);
      $avgSort         = checkError($avgSort);
      
      $avgCSRScalar    = checkError($avgCSRScalar);
      $avgCSRVector    = checkError($avgCSRVector);
      $avgEllpack      = checkError($avgEllpack);
      $avgCSRScalar_DP = checkError($avgCSRScalar_DP);
      $avgCSRVector_DP = checkError($avgCSRVector_DP);
      $avgEllpack_DP   = checkError($avgEllpack_DP);
      $avgTriad        = checkError($avgTriad);

      # Print Average and Total Measurements
      print "Average Measurements Per Node: \n";
      print "Max FLOPS  (GFLOPS):               " . $avgFlops . "\n";
      print "Memory Bandwidth (GB/s):           " . $avgBandwidth . "\n";
      print "PCIe Bus Speed H->D (GB/s):        " . $avgBusD . "\n";
      print "PCIe Bus Speed D->H (GB/s):        " . $avgBusR . "\n";
      if ( !$useCuda ) {
         print "Kernel Compilation (s):            " . $avgKernel . "\n";
         print "OCL Queueing Delay (ms):           " . $avgDelay . "\n";
      }
      print "\nTotals: \n";
      print "Max FLOPS  (GFLOPS):               "
        . $avgFlops * $totalRanks . "\n";
      print "Memory Bandwidth (GB/s):           "
        . $avgBandwidth * $totalRanks . "\n";
      print "PCIe Bus Speed H->D (GB/s):        "
        . $avgBusD * $totalRanks . "\n";
      print "PCIe Bus Speed D->H (GB/s):        "
        . $avgBusR * $totalRanks . "\n";
      if ( !$useCuda ) {
         print "Kernel Compilation (s):          "
          . $avgKernel * $totalRanks . "\n";
      print "OCL Queueing Delay (ms):         "
          . $avgDelay * $totalRanks . "\n\n";
      }
      else {
         print "\n";
      }

      print "- Level 1: Low Level Operations -\n";

      print "Average Measurements Per Node: \n";
      print "SP-FFT (GFLOPS):        " . $avgSpFFT . "\n";
      print "DP-FFT (GFLOPS):        " . $avgDpFFT . "\n";
      print "MD (GFLOPS):            " . $avgMD . "\n";
      print "MD-DP (GFLOPS):         " . $avgMD_DP . "\n";
      print "Reduction (GB/s):       " . $avgReduction . "\n";
      print "Reduction_DP (GB/s):    " . $avgReduction_DP . "\n";
      print "Scan (GB/s):            " . $avgScan . "\n";
      print "Scan-DP (GB/s):         " . $avgScan_DP . "\n";
      print "SGEMM (GFLOPS):         " . $avgSGEMM . "\n";
      print "DGEMM (GFLOPS):         " . $avgDGEMM . "\n";
      print "Sort (GB/s):            " . $avgSort . "\n";
      print "SpMV\n";
      print "CSR-Scalar (GFLOPS):    " . $avgCSRScalar . "\n";
      print "CSR-Scalar-DP (GFLOPS): " . $avgCSRScalar_DP . "\n";
      print "CSR-Vector (GFLOPS):    " . $avgCSRVector . "\n";
      print "CSR-Vector-DP (GFLOPS): " . $avgCSRVector_DP . "\n";
      print "ELLPACK-R  (GFLOPS):    " . $avgEllpack . "\n";
      print "ELLPACK-R-DP  (GFLOPS): " . $avgEllpack_DP . "\n";
      print "Triad (GB/s):           " . $avgTriad . "\n\n";

      print "Totals: \n";
      print "FFT (GFLOPS):           " . $avgSpFFT * $totalRanks . "\n";
      print "FFT-DP (GFLOPS):        " . $avgDpFFT * $totalRanks . "\n";
      print "MD (GFLOPS):            " . $avgMD * $totalRanks . "\n";
      print "MD-DP (GFLOPS):         " . $avgMD_DP * $totalRanks . "\n";
      print "Reduction (GB/s):       " . $avgReduction * $totalRanks . "\n";
      print "Reduction-DP (GB/s):    " . $avgReduction_DP * $totalRanks . "\n";
      print "Scan (GB/s):            " . $avgScan * $totalRanks . "\n";
      print "Scan-DP (GB/s):         " . $avgScan_DP * $totalRanks . "\n";
      print "SGEMM (GFLOPS):         " . $avgSGEMM * $totalRanks . "\n";
      print "DGEMM (GFLOPS):         " . $avgDGEMM * $totalRanks . "\n";
      print "Sort (GB/s):            " . $avgSort * $totalRanks . "\n";
      print "SpMV\n";
      print "CSR-Scalar (GFLOPS):    " . $avgCSRScalar * $totalRanks . "\n";
      print "CSR-Scalar-DP (GFLOPS): " . $avgCSRScalar_DP * $totalRanks . "\n";
      print "CSR-Vector (GFLOPS):    " . $avgCSRVector * $totalRanks . "\n";
      print "CSR-Vector-DP (GFLOPS): " . $avgCSRVector_DP * $totalRanks . "\n";
      print "ELLPACK-R  (GFLOPS):    " . $avgEllpack * $totalRanks . "\n";
      print "ELLPACK-R-DP  (GFLOPS): " . $avgEllpack_DP * $totalRanks . "\n";
      print "Triad (GB/s):        " . $avgTriad * $totalRanks . "\n\n";
 
      print "- Level 2: Application Kernels -\n";
    
      print "Average Measurements Per Node: \n";
      print "S3D    (GFLOPS):            " . $avgS3D . "\n";
      print "S3D-DP (GFLOPS):            " . $avgS3D_DP . "\n";
    
      print "Totals: \n";
      print "S3D    (GFLOPS):            " . $avgS3D * $totalRanks . "\n";
      print "S3D-DP (GFLOPS):            " . $avgS3D_DP * $totalRanks . "\n";
      # Also output the results to a .csv file for analysis

      # Print Header
      print OUTFILE "Device Name, Max FLOPS, Device Bandwidth, PCIe H->D, ";
      print OUTFILE "PCIe D->H, ";
      if ( !$useCuda ) {
         print OUTFILE "kCompile, QDelay, ";
      }
      print OUTFILE "FFT, FFT-DP, MD, MD-DP, Reduction, Reduction-DP, Scan, ";
      print OUTFILE "Scan-DP, SGEMM, DGEMM, Sort, Triad\n";

      # Print Avg Results
      print OUTFILE
        "avgPerDevice, $avgFlops, $avgBandwidth, $avgBusD, $avgBusR";
      if ( !$useCuda ) {
         print OUTFILE ", $avgKernel, $avgDelay";
      }
      print OUTFILE ", $avgSpFFT, $avgDpFFT, $avgMD, $avgMD_DP, $avgReduction, ";
      print OUTFILE "$avgReduction_DP, $avgScan, $avgScan_DP, $avgSGEMM, ";
      print OUTFILE "$avgDGEMM, $avgSort, $avgTriad\n";

      # Print Total Results
      print OUTFILE "totals, ";
      print OUTFILE $avgFlops * $totalRanks,     ", ";
      print OUTFILE $avgBandwidth * $totalRanks, ", ";
      print OUTFILE $avgBusD * $totalRanks,      ", ";
      print OUTFILE $avgBusR * $totalRanks,      ", ";
      if ( !$useCuda ) {
         print OUTFILE $avgKernel * $totalRanks, ", ";
         print OUTFILE $avgDelay * $totalRanks,  ", ";
      }
      print OUTFILE $avgSpFFT * $totalRanks,       ", ";
      print OUTFILE $avgDpFFT * $totalRanks,       ", ";
      print OUTFILE $avgMD * $totalRanks,        ", ";
      print OUTFILE $avgMD_DP * $totalRanks,        ", ";
      print OUTFILE $avgReduction * $totalRanks, ", ";
      print OUTFILE $avgReduction_DP * $totalRanks, ", ";
      print OUTFILE $avgScan * $totalRanks,      ", ";
      print OUTFILE $avgScan_DP * $totalRanks,      ", ";
      print OUTFILE $avgSGEMM * $totalRanks,     ", ";
      print OUTFILE $avgDGEMM * $totalRanks,     ", ";
      print OUTFILE $avgSort * $totalRanks,      ", ";
      print OUTFILE $avgTriad * $totalRanks,     "\n";
   }
   close(OUTFILE);
   print "Results written to: results.csv \n";
}

# Subroutine serialBM(numDevices)
# Purpose: Runs Serial Benchmarks
# Arg 1: numDevices: Number of available devices
# Arg 2: useCuda: flag to use cuda or opencl
sub serialBM {
   my $numDevices = $_[0];
   my $useCuda    = $_[1];
   my $sizeClass  = getArg("-s");
   my $i;
   my @devNames;
   open( OUTFILE, ">results.csv" ) or die $!;

   # Print Header
   print OUTFILE "Results from the SHOC Benchmark Suite Version 1.0\n";
   print OUTFILE "Number of Devices, $_[0] \n";
   for ( $i = 0 ; $i < $_[0] ; $i++ ) {
      $devNames[$i] = getDeviceName( $useCuda, $i );
      print OUTFILE "Device $i, " . $devNames[$i] . "\n";
   }

   # Run all the level 0 benchmarks, and store their output in log files   
   print "- Level 0: \"Feeds and Speeds\" -\n";
   print "-- This can take several minutes. --\n";

   # ====================
   # PCIe Bandwidth Tests
   # ====================
   # Print results file header
   print OUTFILE "PCIe Bandwidth Test\n";
   print OUTFILE "Sizes, ";
   print "-PCIe Bandwidth Tests (GB/s)-\n";

   # Write out full bandwidth chart to results.csv file
   my $numSizes = 17;    # 17 sizes starting at 1 kb
   my @down_bandwidths;
   my @up_bandwidths;
   my $sizeCounter;
   my $currSize = 1;
   for ( $sizeCounter = 0 ; $sizeCounter < $numSizes ; $sizeCounter++ ) {
      print OUTFILE "$currSize" . "kB, ";
      $currSize = $currSize * 2;
   }

   for ( $i = 0 ; $i < $numDevices ; $i++ ) {
      my $currDeviceName = getDeviceName( $useCuda, $i );      
      system( buildCommand( "BusSpeedDownload", $i ) );
      system( buildCommand( "BusSpeedReadback", $i ) );

      # Grep for max results
      my $avgBusD =
        findmax( buildFileName( "BusSpeedDownload", $i ), "DownloadSpeed" );
      my $avgBusR =
        findmax( buildFileName( "BusSpeedReadback", $i ), "ReadbackSpeed" );
      print "Dev $i: $devNames[$i] H->D: " . $avgBusD . "\n";
      print "Dev $i: $devNames[$i] D->H: " . $avgBusR . "\n";

      $currSize = 1;
      print OUTFILE "\n$i: $devNames[$i] H->D, ";
      for ( $sizeCounter = 0 ; $sizeCounter < $numSizes ; $sizeCounter++ ) {
         print OUTFILE findmax_attr( buildFileName( "BusSpeedDownload", $i ),
            "DownloadSpeed", "$currSize" . "kB" )
           . ", ";
         $currSize = $currSize * 2;
      }
      $currSize = 1;
      print OUTFILE "\n$devNames[$i] D->H, ";
      for ( $sizeCounter = 0 ; $sizeCounter < $numSizes ; $sizeCounter++ ) {
         print OUTFILE findmax_attr( buildFileName( "BusSpeedReadback", $i ),
            "ReadbackSpeed", "$currSize" . "kB" )
           . ", ";
         $currSize = $currSize * 2;
      }
   }

   # ====================
   # MaxFlops Test
   # ====================
   # Print header to console
   print "\n-MaxFlops Test (GFLOPS)-\n";

   # Write out header to results.csv
   print OUTFILE "\n\nMax FLOPS Benchmark (GFLOPS) \n";
   print OUTFILE "Tests, SP, DP\n";

   for ( $i = 0 ; $i < $numDevices ; $i++ ) {            
      system( buildCommand( "MaxFlops", $i ) );
      my $avgSPFlops;
      my $avgDPFlops;
      if ($useCuda) {
         $avgSPFlops = findanymax( buildFileName( "MaxFlops", $i ), "-SP" );
         $avgDPFlops = findanymax( buildFileName( "MaxFlops", $i ), "-DP" );
      }
      else {
         $avgSPFlops = findanymax( buildFileName( "MaxFlops", $i ), "-SP" );
         $avgDPFlops = findanymax( buildFileName( "MaxFlops", $i ), "-DP" );
      }
      # Print max FLOPS rate to console
      print "Dev $i: $devNames[$i] SP:   " . $avgSPFlops . "\n";
      print "Dev $i: $devNames[$i] DP:   " . $avgDPFlops . "\n";
      # Write results to file
      print OUTFILE "$devNames[$i], $avgSPFlops, $avgDPFlops \n";      
   }
   # ====================
   # DeviceMemory Test
   # ====================
   
   # Print results to console
   print "\n-Device Memory Bandwidth Tests (GB/s) (Read / Write)-\n";
   print OUTFILE "\nDeviceMemory Bandwidth Benchmark (GB/s) \n";
   if ($useCuda) {
      print OUTFILE "Test Name, Direction, Global (contiguous), Global (Strided), Shared, Texture\n";
   }
   else {
      print OUTFILE "Test Name, Direction, Global (contiguous), Global (Strided), Local, Image\n";
   }
       
   for ( $i = 0 ; $i < $numDevices ; $i++ ) {      
      # Execute the benchmark
      system( buildCommand( "DeviceMemory", $i ) );

      # Grep Read Bandwidths
      my $gReadCoa  = findmax( buildFileName( "DeviceMemory", $i ), "readGlobalMemoryCoalesced" );
      my $gReadUnit = findmax( buildFileName( "DeviceMemory", $i ), "readGlobalMemoryUnit" );
      my $sRead     =   findmax( buildFileName( "DeviceMemory", $i ), "readLocalMemory" );
      my $texRand;
      if ($useCuda) {
          $texRand = findmax( buildFileName( "DeviceMemory", $i ),  "TextureRepeatedRandomAccess");
      }
      else {
        $texRand = findmax( buildFileName( "DeviceMemory", $i ), "ImageRandAccess" );
      }
      # Grep Write Bandwidths
      my $gWriteCoa  = findmax( buildFileName( "DeviceMemory", $i ),"writeGlobalMemoryCoalesced" );
      my $gWriteUnit = findmax( buildFileName( "DeviceMemory", $i ), "writeGlobalMemoryUnit" );
      my $sWrite     = findmax( buildFileName( "DeviceMemory", $i ), "writeLocalMemory" );

      print "Dev $i: $devNames[$i]\n";
      print "Global Memory Contiguous:       $gReadCoa / $gWriteCoa \n";
      print "Global Memory Strided:          $gReadUnit / $gWriteUnit \n";
      if ($useCuda) {
          print "Shared Memory:                  $sRead / $sWrite \n";
      }
      else {
         print "Local Memory:                   $sRead / $sWrite \n";
      }
      if ($useCuda) {
         print "Texture (Random Access):        $texRand \n";
      }
      else {
         print "Image (Random Access):          $texRand \n";
      }   
      print OUTFILE "$devNames[$i], Read, $gReadCoa, $gReadUnit, $sRead, $texRand \n";
      print OUTFILE "$devNames[$i], Write, $gWriteCoa, $gWriteUnit, $sWrite \n";
   }
   # ========================
   # KernelCompile and QDelay
   # ========================
   if ( !$useCuda ) {
      # print headers
      print "\n-OpenCL Kernel Compilation (s)-\n";
      print OUTFILE "\nOpenCL Implementation Benchmarks\n";
      print OUTFILE "Test Name, KernelCompile (s)\n";       
      for ( $i = 0 ; $i < $numDevices ; $i++ ) {   
         system( buildCommand( "KernelCompile", $i ) );
         my $avgKernel = findmin( buildFileName( "KernelCompile", $i ), "BuildProgram" );
         print "Dev $i: $devNames[$i] Kernel Compilation:         " . $avgKernel . "\n";
         print OUTFILE "$devNames[$i], $avgKernel\n";  
      }
      print "\n-OpenCL Queuing Delay (ms)-\n";
      print OUTFILE "Test Name, Sub-Start Delay (ms)\n";
      for ( $i = 0 ; $i < $numDevices ; $i++ ) {
         system( buildCommand( "QueueDelay",    $i ) );      
         my $avgDelay = findmin( buildFileName( "QueueDelay", $i ), "SSDelay" );
         print "Dev $i: $devNames[$i] Submit-Start Delay:        " . $avgDelay . "\n";
         print OUTFILE "$devNames[$i], $avgDelay\n";
      }      
   }

   print "\n--- Level 1 - Basic Algorithms and Parallel Primitives ---\n";   
   print "-- This can take several minutes. --\n";
   print OUTFILE "\n\n Level 1 Results\n";   
   #====
   # FFT
   #====
   # print headers
   print "\n-FFT (GFLOPS) (Kernel Only / Kernel + PCIe transfer)-\n";
   print OUTFILE "\nFFT\n";
   print OUTFILE "Test Name, SP FFT, SP IFFT, SP FFT_PCIe, SP IFFT_PCIe, DP FFT, DP IFFT, DP FFT_PCIe, DP IFFT_PCIe\n";
   
   for ( $i = 0 ; $i < $numDevices ; $i++ ) {
      system( buildCommand( "FFT", $i ) );
      my $avgFFT       = findmax( buildFileName( "FFT", $i ), "SP-FFT" );
      my $avgFFT_PCIe  = findmax( buildFileName( "FFT", $i ), "SP-FFT_PCIe" );
      my $avgIFFT      = findmax( buildFileName( "FFT", $i ), "SP-FFT-INV" );
      my $avgIFFT_PCIe = findmax( buildFileName( "FFT", $i ), "SP-FFT-INV_PCIe" );

      my $avgFFT_DP       = findmax( buildFileName( "FFT", $i ), "DP-FFT" );
      my $avgFFT_DP_PCIe  = findmax( buildFileName( "FFT", $i ), "DP-FFT_PCIe" );
      my $avgIFFT_DP      = findmax( buildFileName( "FFT", $i ), "DP-FFT-INV" );
      my $avgIFFT_DP_PCIe = findmax( buildFileName( "FFT", $i ), "DP-FFT-INV_PCIe" );

      # Write results to console   
      print "Dev $i: $devNames[$i] SP FFT:           " . $avgFFT     . " / "  . $avgFFT_PCIe     . "\n";
      print "Dev $i: $devNames[$i] SP IFFT+Norm:     " . $avgIFFT    . " / "  . $avgIFFT_PCIe    . "\n";   
      print "Dev $i: $devNames[$i] DP FFT:           " . $avgFFT_DP  . " / "  . $avgFFT_DP_PCIe  . "\n";
      print "Dev $i: $devNames[$i] DP IFFT+Norm:     " . $avgIFFT_DP . " / "  . $avgIFFT_DP_PCIe . "\n";

      # Write results to file
      print OUTFILE "$devNames[$i], $avgFFT, $avgIFFT, $avgFFT_PCIe, $avgIFFT_PCIe, ";
      print OUTFILE "$avgFFT_DP, $avgIFFT_DP, $avgFFT_DP_PCIe, $avgIFFT_DP_PCIe\n";
   }

   ###########
   # GEMM
   ###########
   print "\n-GEMM (GFLOPS/s) (Kernel Only / Kernel + PCIe transfer)-\n";
   print OUTFILE "\nGEMM\n";
   print OUTFILE "Test Name, SGEMM (GFLOPS), SGEMM_PCIe (GFLOPS), DGEMM (GFLOPS), DGEMM_PCIe (GFLOPS)\n";
   for ( $i = 0 ; $i < $numDevices ; $i++ ) {
      system( buildCommand( "SGEMM", $i ) );
      my $GEMM         = findmax( buildFileName( "SGEMM", $i ), "SGEMM-N" );
      my $GEMM_PCIe    = findmax( buildFileName( "SGEMM", $i ), "SGEMM-N_PCIe" );
      my $GEMM_DP      = findmax( buildFileName( "SGEMM", $i ), "DGEMM-N" );
      my $GEMM_PCIe_DP = findmax( buildFileName( "SGEMM", $i ), "DGEMM-N_PCIe" );
        
      print "Dev $i: $devNames[$i] SGEMM:           " . $GEMM    . " / "  . $GEMM_PCIe    . "\n";
      print "Dev $i: $devNames[$i] DGEMM:           " . $GEMM_DP . " / "  . $GEMM_PCIe_DP . "\n";        
      print OUTFILE "$devNames[$i], $GEMM, $GEMM_PCIe, $GEMM_DP, $GEMM_PCIe_DP \n";        
   }

   ####
   # MD
   ####
   print "\n-MD (GB/s) (Kernel Only / Kernel + PCIe transfer)-\n";
   print OUTFILE "\nMD\n";
   print OUTFILE "Test Name, SP LJ (GB/s), SP LJ_PCIe (GB/s), DP LJ (GB/s), DP LJ_PCIe (GB/s)\n";
   for ( $i = 0 ; $i < $numDevices ; $i++ ) {         
      system( buildCommand( "MD",        $i ) );
      my $avgLJ         = findmax( buildFileName( "MD", $i ), "MD-LJ-Bandwidth" );
      my $avgLJ_PCIe    = findmax( buildFileName( "MD", $i ), "MD-LJ-Bandwidth_PCIe" );
      my $avgLJ_DP      = findmax( buildFileName( "MD", $i ), "MD-LJ-DP-Bandwidth" );
      my $avgLJ_PCIe_DP = findmax( buildFileName( "MD", $i ), "MD-LJ-DP-Bandwidth_PCIe" );
        
      print "Dev $i: $devNames[$i] SP MD:           " . $avgLJ    . " / "  . $avgLJ_PCIe    . "\n";
      print "Dev $i: $devNames[$i] DP MD:           " . $avgLJ_DP . " / "  . $avgLJ_PCIe_DP . "\n";        
      print OUTFILE "$devNames[$i], $avgLJ, $avgLJ_PCIe, $avgLJ_DP, $avgLJ_PCIe_DP \n";        
   }
   ###########
   # Reduction
   ###########
   print "\n-Reduction (GB/s) (Kernel Only / Kernel + PCIe transfer)-\n";
   print OUTFILE "\nReduction\n";
   print OUTFILE "Test Name, SP Reduction (GB/s), SP Reduction_PCIe (GB/s), DP Reduction (GB/s), DP Reduction_PCIe (GB/s)\n";
   for ( $i = 0 ; $i < $numDevices ; $i++ ) {
      system( buildCommand( "Reduction",        $i ) );
      my $reduce         = findmax( buildFileName( "Reduction", $i ), "Reduction" );
      my $reduce_PCIe    = findmax( buildFileName( "Reduction", $i ), "Reduction_PCIe" );
      my $reduce_DP      = findmax( buildFileName( "Reduction", $i ), "Reduction-DP" );
      my $reduce_PCIe_DP = findmax( buildFileName( "Reduction", $i ), "Reduction-DP_PCIe" );
        
      print "Dev $i: $devNames[$i] SP Reduction:           " . $reduce    . " / "  . $reduce_PCIe    . "\n";
      print "Dev $i: $devNames[$i] DP Reduction:           " . $reduce_DP . " / "  . $reduce_PCIe_DP . "\n";        
      print OUTFILE "$devNames[$i], $reduce, $reduce_PCIe, $reduce_DP, $reduce_PCIe_DP \n";        
  }
   
   ###########
   # S3D
   ###########
   print "\n-S3D (GFLOPS) (Kernel Only / Kernel + PCIe transfer)-\n";
   print OUTFILE "\nS3D\n";
   print OUTFILE "Test Name, S3D (GFLOPS), S3D_PCIe (GFLOPS), DP S3D (GFLOPS), DP S3D_PCIe (GFLOPS)\n";
   for ( $i = 0 ; $i < $numDevices ; $i++ ) {
      system( buildCommand( "S3D",        $i ) );
      my $s3d         = findmax( buildFileName( "S3D", $i ), "S3D-SP" );
      my $s3d_PCIe    = findmax( buildFileName( "S3D", $i ), "S3D-SP_PCIe" );
      my $s3d_DP      = findmax( buildFileName( "S3D", $i ), "S3D-DP" );
      my $s3d_PCIe_DP = findmax( buildFileName( "S3D", $i ), "S3D-DP_PCIe" );
        
      print "Dev $i: $devNames[$i] SP S3D:           " . $s3d    . " / "  . $s3d_PCIe    . "\n";
      print "Dev $i: $devNames[$i] DP S3D:           " . $s3d_DP . " / "  . $s3d_PCIe_DP . "\n";        
      print OUTFILE "$devNames[$i], $s3d, $s3d_PCIe, $s3d_DP, $s3d_PCIe_DP \n";
   }
    
   ###########
   # Scan
   ###########
   print "\n-Scan (GB/s) (Kernel Only / Kernel + PCIe transfer)-\n";
   print OUTFILE "\nScan\n";
   print OUTFILE "Test Name, SP Scan (GB/s), SP Scan_PCIe (GB/s), DP Scan (GB/s), DP Scan_PCIe (GB/s)\n";
   for ( $i = 0 ; $i < $numDevices ; $i++ ) {
      system( buildCommand( "Scan",        $i ) );
      my $scan         = findmax( buildFileName( "Scan", $i ), "Scan" );
      my $scan_PCIe    = findmax( buildFileName( "Scan", $i ), "Scan_PCIe" );
      my $scan_DP      = findmax( buildFileName( "Scan", $i ), "Scan-DP" );
      my $scan_PCIe_DP = findmax( buildFileName( "Scan", $i ), "Scan-DP_PCIe" );
        
      print "Dev $i: $devNames[$i] SP Scan:           " . $scan    . " / "  . $scan_PCIe    . "\n";
      print "Dev $i: $devNames[$i] DP Scan:           " . $scan_DP . " / "  . $scan_PCIe_DP . "\n";        
      print OUTFILE "$devNames[$i], $scan, $scan_PCIe, $scan_DP, $scan_PCIe_DP \n";        
   }

   ###########
   # Sort
   ###########
   print "\n-Sort (GB/s) (Kernel Only / Kernel + PCIe transfer)-\n";
   print OUTFILE "\nSort\n";
   print OUTFILE "Test Name, SP Sort (GB/s), SP Sort_PCIe (GB/s)\n";
   for ( $i = 0 ; $i < $numDevices ; $i++ ) {
      system( buildCommand( "Sort",        $i ) );
      my $Sort         = findmax( buildFileName( "Sort", $i ), "Sort-Rate" );
      my $Sort_PCIe    = findmax( buildFileName( "Sort", $i ), "Sort-Rate_PCIe" );         
      print "Dev $i: $devNames[$i] Sort:            " . $Sort    . " / "  . $Sort_PCIe    . "\n";
      print OUTFILE "$devNames[$i], $Sort, $Sort_PCIe\n";        
   }
  
   ###########
   # Spmv
   ###########
   print "\n-Sparse Matrix-Vector Multiply (SpMV) (GFLOPS) (Kernel Only / Kernel + PCIe transfer)-\n";
   print OUTFILE "\nSpMV\n";
   print OUTFILE "Test Name, CSR-Scalar, Padded CSR-Scalar, CSR-Vector, Padded CSR-Vector, ELLPACKR, ";
   print OUTFILE "CSR-Scalar_PCIe, Padded CSR-Scalar_PCIe, CSR-Vector_PCIe, Padded CSR-Vector_PCIe, ";
   print OUTFILE "CSR-Scalar-DP, Padded CSR-Scalar-DP, CSR-Vector-DP, Padded CSR-Vector-DP, ELLPACKR-DP, ";
   print OUTFILE "CSR-Scalar-DP_PCIe, Padded CSR-Scalar-DP_PCIe, CSR-Vector-DP_PCIe, Padded CSR-Vector-DP-PCIe, \n";
   for ( $i = 0 ; $i < $numDevices ; $i++ ) {
      system( buildCommand( "Spmv",        $i ) );
      my $csrScalar           = findmax( buildFileName( "Spmv", $i ), "CSR-Scalar-SP" );
      my $csrScalar_PCIe      = findmax( buildFileName( "Spmv", $i ), "CSR-Scalar-SP_PCIe" );
      my $csrScalar_DP        = findmax( buildFileName( "Spmv", $i ), "CSR-Scalar-DP" );
      my $csrScalar_DP_PCIe   = findmax( buildFileName( "Spmv", $i ), "CSR-Scalar-DP_PCIe" );
        
      my $pcsrScalar          = findmax( buildFileName( "Spmv", $i ), "Padded_CSR-Scalar-SP" );
      my $pcsrScalar_PCIe     = findmax( buildFileName( "Spmv", $i ), "Padded_CSR-Scalar-SP_PCIe" );
      my $pcsrScalar_DP       = findmax( buildFileName( "Spmv", $i ), "Padded_CSR-Scalar-DP" );
      my $pcsrScalar_DP_PCIe  = findmax( buildFileName( "Spmv", $i ), "Padded_CSR-Scalar-DP_PCIe" );

      my $csrVector           = findmax( buildFileName( "Spmv", $i ), "CSR-Vector-SP" );
      my $csrVector_PCIe      = findmax( buildFileName( "Spmv", $i ), "CSR-Vector-SP_PCIe" );
      my $csrVector_DP        = findmax( buildFileName( "Spmv", $i ), "CSR-Vector-DP" );
      my $csrVector_DP_PCIe   = findmax( buildFileName( "Spmv", $i ), "CSR-Vector-DP_PCIe" );
        
      my $pcsrVector          = findmax( buildFileName( "Spmv", $i ), "Padded_CSR-Vector-SP" );
      my $pcsrVector_PCIe     = findmax( buildFileName( "Spmv", $i ), "Padded_CSR-Vector-SP_PCIe" );
      my $pcsrVector_DP       = findmax( buildFileName( "Spmv", $i ), "Padded_CSR-Vector-DP" );
      my $pcsrVector_DP_PCIe  = findmax( buildFileName( "Spmv", $i ), "Padded_CSR-Vector-DP_PCIe" );

      my $ellpack             = findmax( buildFileName( "Spmv", $i ), "ELLPACKR-SP" );
      my $ellpack_DP          = findmax( buildFileName( "Spmv", $i ), "ELLPACKR-DP" );


      print "Dev $i: $devNames[$i] CSR-Scalar:           " . $csrScalar    . " / "  . $csrScalar_PCIe  . "\n";
      print "Dev $i: $devNames[$i] Padded CSR-Scalar:    " . $pcsrScalar   . " / "  . $pcsrScalar_PCIe . "\n";
      
      print "Dev $i: $devNames[$i] CSR-Vector:           " . $csrVector    . " / "  . $csrVector_PCIe  . "\n";
      print "Dev $i: $devNames[$i] Padded CSR-Vector:    " . $pcsrVector   . " / "  . $pcsrVector_PCIe . "\n";

      print "Dev $i: $devNames[$i] DP CSR-Scalar:        " . $csrScalar_DP    . " / "  . $csrScalar_DP_PCIe  . "\n";
      print "Dev $i: $devNames[$i] Padded DP CSR-Scalar: " . $pcsrScalar_DP   . " / "  . $pcsrScalar_DP_PCIe . "\n";
      
      print "Dev $i: $devNames[$i] DP CSR-Vector:        " . $csrVector_DP  . " / "  . $csrVector_DP_PCIe  . "\n";
      print "Dev $i: $devNames[$i] Padded DP CSR-Vector: " . $pcsrVector_DP . " / "  . $pcsrVector_DP_PCIe . "\n";
 
      print "Dev $i: $devNames[$i] SP ELLPACKR:          " . $ellpack    . "\n";
      print "Dev $i: $devNames[$i] DP ELLPACKR:          " . $ellpack_DP    . "\n";
      
      
      print OUTFILE "$devNames[$i], $csrScalar, $pcsrScalar, $csrVector, $pcsrVector, $ellpack, ";
      print OUTFILE "$csrScalar_PCIe, $pcsrScalar_PCIe, $csrVector_PCIe, $pcsrVector_PCIe, ";
      print OUTFILE "$csrScalar_DP, $pcsrScalar_DP, $csrVector_DP, $pcsrVector_DP, $ellpack_DP, ";
      print OUTFILE "$csrScalar_DP_PCIe, $pcsrScalar_DP_PCIe, $csrVector_DP_PCIe, $pcsrVector_DP_PCIe \n";
 
   }

   ###########
   # Stencil2D
   ###########
   print "\n-Stencil2D (s) (Kernel + PCIe transfer)-\n";
   print OUTFILE "\nStencil2D\n";
   print OUTFILE "Test Name, Stencil2D (s)\n";
   for ( $i = 0 ; $i < $numDevices ; $i++ ) {
      system( buildCommand( "Stencil2D",        $i ) );
      my $sten = findmin( buildFileName( "Stencil2D", $i ), "SP_Sten2D" );                
      my $stendp = findmin( buildFileName( "Stencil2D", $i ), "DP_Sten2D" );                
      print "Dev $i: $devNames[$i] SP Sten2D:        " . $sten . "\n";                
      print "Dev $i: $devNames[$i] DP Sten2D:        " . $stendp . "\n";                
      print OUTFILE "$devNames[$i], $sten, $stendp \n";        
   }   
   
   ###########
   # Triad
   ###########
   print "\n-Triad (GB/s) (Kernel + PCIe transfer)-\n";
   print OUTFILE "\nTriad\n";
   print OUTFILE "Test Name, Triad (GB/s)\n";
   for ( $i = 0 ; $i < $numDevices ; $i++ ) {
      system( buildCommand( "Triad",        $i ) );
      my $triad = findmax( buildFileName( "Triad", $i ), "TriadBdwth" );                
      print "Dev $i: $devNames[$i] Triad:           " . $triad . "\n";                
      print OUTFILE "$devNames[$i], $triad \n";        
   }   

   close(OUTFILE);
   print "Results written to: results.csv \n";
}

# Subroutine: buildParCommand(testName, totalRanks)
# Purpose: Helper routine to construct commands to run benchmarks
# testName: name of the benchmark
# totalRanks: Number of MPI ranks to launch

sub buildParCommand {
   my $prog      = $_[0];
   my $np        = $_[1];
   my $devPerNd  = $_[2];
   my $devArg    = $_[3];
   my $useCuda   = getStringArg("-cuda");
   my $sizeClass = getArg("-s");
   my $platformString;
   if ($useCuda) {
      $platformString = "CUDA/";
   }
   else {
      $platformString = "OpenCL/";
   }
  
  # If the user specified a host file, pass that to mpirun
   my $hostfile  = getArg("-h");
   my $hostfileString;
   if ($hostfile eq "none") {
      $hostfileString = " ";
   } else {
      $hostfileString = "-hostfile $hostfile ";
   }

   my $str;
   if (getArg("-read-only")) {
       $str = "echo " . $prog;
   }
   else {
      $str = "mpirun -np $np $hostfileString "
      # . " -npernode $devPerNd "
     . $bindir . "/EP/$platformString" 
     . $prog
     . " -s $sizeClass $devArg >"
     . buildFileName( $prog, 0 ) . " 2>"
     . buildFileName( $prog, 0 ) . ".err";
   }
   #    print "Built command: $str \n";

   return $str;
}

# Subroutine: buildCommand(testName, deviceNum)
# Purpose: Helper routine to construct commands to run benchmarks
# testName -- name of test
# deviceNum -- device number

sub buildCommand {
   my $useCuda = getStringArg("-cuda");
   my $platformString;
   my $sizeClass = getArg("-s");
   if ($useCuda) {
      $platformString = "CUDA/";
   }
   else {
      $platformString = "OpenCL/";
   }
   my $str;
   if (getArg("-read-only")) {
       $str = "echo " . $_[0];
   }
   else {
       $str = $bindir . "/Serial/"
     . $platformString
     . $_[0]
     . " -s $sizeClass -d "
     . $_[1] . " >"
     . buildFileName( $_[0], $_[1] ) . " 2>"
     . buildFileName( $_[0], $_[1] ) . ".err";
   }
     # print "Built command: $str \n";
   return $str;
}

# Subroutine: buildFileName(testName, deviceNum)
# Purpose: Helper routine to construct fileNames
# testName -- name of test
# deviceNum -- device number
sub buildFileName {
   return "Logs/dev" . $_[1] . "_" . $_[0] . ".log";
}

# Subroutine: printDevInfo
# Purpose: Print info about available devices
sub printDevInfo {
   my $useCuda = $_[0];
   my $devNameString;
   my $retval;
   my $devNumString = "Number of devices";

   # If the user specified a host file, pass that to mpirun
   my $hostfile  = getArg("-h");
   my $hostfileString;
   if ($hostfile eq "none") {
    $hostfileString = "";
   } else {
    $hostfileString = "mpirun -np 1 -hostfile $hostfile ";
   }

   # Run a level 0 benchmark with the device info flag, and
   # figure out the number of available devices
   if (!$readonly) {
       if ($useCuda) {
          $devNameString = "name";
          $retval        = system($hostfileString . $bindir . 
          "/Serial/CUDA/BusSpeedDownload -i > Logs/deviceInfo.txt 2> Logs/deviceInfo.err" );
       }
       else {
          $devNameString = "DeviceName";
          $retval        = system($hostfileString . $bindir . 
          "/Serial/OpenCL/BusSpeedDownload -i > Logs/deviceInfo.txt 2> Logs/deviceInfo.err");
       }

       die "Error collecting device info (need to set -bin-dir?)\n" unless $retval == 0;
   }

   # Now parse the device info, and figure out how many devices are available
   open( INFILE, "./Logs/deviceInfo.txt" );

   my $line;
   my @tokens;
   my $n;
   my @deviceNames;
   while ( $line = <INFILE> ) {
      chomp($line);
      $line =~ s/^\s+//;    #remove leading spaces
      @tokens = split( /\s+=\s+/, $line );
      if ( $tokens[0] eq $devNumString ) {
         $n = int( $tokens[1] );
      }
      if ( $tokens[0] eq $devNameString ) {
         push( @deviceNames, $tokens[1] );
      }
   }
   close(INFILE);
   print "Number of available devices: $n \n";
   my $i;
   for ( $i = 0 ; $i < $n ; $i++ ) {
      print "Device $i: ", $deviceNames[$i], "\n";
   }
   return $n;
}

# Subroutine: getArg(numArgs, arg)
# Purpose: Parse command line arguments and find number of nodes (default 1)
#          Assumes all arguments are integers
# numArgs -- the number of arguments
# arg -- the argument (i.e. -n)
sub getArg() {
   my $numArgs = $#ARGV + 1;
   my $n;
   foreach $n ( 0 .. $numArgs ) {
      if ( $ARGV[$n] eq $_[0] ) {
         usage()
           unless ( ( $numArgs > $n + 1 )
#            && ( isInt( $ARGV[ $n + 1 ] ) ) 
               );
         return $ARGV[ $n + 1 ];
      }
   }

   # Argument wasn't found, return defaults
   if ( $_[0] eq "-n" || $_[0] eq "-d" ) {
      return 1;
   }
   if ( $_[0] eq "-bin-dir") {
       return "../bin";
   }
   if ( $_[0] eq "-h") {
       return "none";
   }
}

# Subroutine: getStringArg(numArgs, arg)
# Purpose: Parse command line arguments and tests to see if a string is present
# numArgs -- the number of arguments
# arg -- the argument (i.e. -n)
sub getStringArg() {
   my $numArgs = $#ARGV + 1;
   my $n;
   foreach $n ( 0 .. $numArgs ) {
      if ( $ARGV[$n] eq $_[0] ) {
         return 1;
      }
   }

   # Argument wasn't found, return false
   return 0;
}
# Subroutine: getDeviceName ($useCuda, $i)
# Purpose: Grep for device i's name
sub getDeviceName {
   my $useCuda = $_[0];
   my $i       = $_[1];
   my $devNameString;
   my $devNumString = "Number of devices";

   if ($useCuda) {
      $devNameString = "name";
   }
   else {
      $devNameString = "DeviceName";
   }

   # Parse the device info, and return the name of device i
   open( INFILE, "./Logs/deviceInfo.txt" );
   my $line;
   my @tokens;
   my $n;
   my @deviceNames;
   while ( $line = <INFILE> ) {
      chomp($line);
      $line =~ s/^\s+//;    #remove leading spaces
      @tokens = split( /\s+=\s+/, $line );
      if ( $tokens[0] eq $devNumString ) {
         $n = int( $tokens[1] );
      }
      if ( $tokens[0] eq $devNameString ) {
         push( @deviceNames, $tokens[1] );
      }
   }
   close(INFILE);
   return $deviceNames[$i];
}

# Subroutine: findmin(fileName, testName)
# Purpose: Parses resultDB output to find minimum value for a specified test
# fileName -- name of log file to open
# testName -- name of test to look for
sub findmin {

   my $filename = $_[0];
   my $testname = $_[1];

   open( LOGFILE, $filename );
#   my $best = 10000000;    # Arbitrary Large Number
   my $best = 1E+37;    # Arbitrary Large Number

   my $line;
   my @tokens;

   while ( $line = <LOGFILE> ) {
      chomp($line);
      $line =~ s/^\s+//;    #remove leading spaces
      @tokens = split( /\s+/, $line );
      if ( $tokens[0] eq $testname ) {
         if ( $tokens[6] < $best ) {    # min is the 6th column
            $best = $tokens[6];
         }
      }
   }
   close(LOGFILE);
   return checkError($best);
}

# Subroutine: findmax(fileName, testName)
# Purpose: Parses resultDB output to find maximum value for a specified test
# fileName -- name of log file to open
# testName -- name of test to look for
#sub findmax {
#
#   my $filename = $_[0];
#   my $testname = $_[1];
#
#   open( LOGFILE, $filename );
#   my $best = -1;
#
#   my $line;
#   my @tokens;
#
#   while ( $line = <LOGFILE> ) {
#      chomp($line);
#      $line =~ s/^\s+//;    #remove leading spaces
#      @tokens = split( /\s+/, $line );
#      if ( $tokens[0] eq $testname ) {
#         if ( $tokens[7] > $best ) {
#            $best = $tokens[7];
#         }
#      }
#   }
#   close(LOGFILE);
#   return checkError($best);
#}


# Subroutine: findmax(fileName, testName)
# Purpose: Parses resultDB output to find maximum value for a specified test
# fileName -- name of log file to open
# testName -- name of test to look for
sub findmax {

   my $filename = $_[0];
   my $testname = $_[1];

   open( LOGFILE, $filename );
   my $best = -1;

   my $line;
   my @tokens;

   while ( $line = <LOGFILE> ) {
      chomp($line);
      $line =~ s/^\s+//;    #remove leading spaces
      @tokens = split( /\s+/, $line );
      if ( $tokens[0] eq $testname ) {
                        # We assume that the 7th column is the max (which could be
                        # nan or inf) and after that there are all the trials.
         for(my $i=7; $i<=$#tokens; $i++){
             if ( $tokens[$i] =~ /inf/ || $tokens[$i] =~ /nan/ ){
                 next;
             }
             if ( $tokens[$i] > $best ) {
                $best = $tokens[$i];
             }
                        }
      }
   }
   close(LOGFILE);
   return checkError($best);
}

# Subroutine: findmax_attr(fileName, testName, attr)
# Purpose: Parses resultDB output to find maximum value for a specified test
# fileName -- name of log file to open
# testName -- name of test to look for
# attr     -- attribute string to match
sub findmax_attr {
   my $filename = $_[0];
   my $testname = $_[1];
   my $attrname = $_[2];
   open( LOGFILE, $filename );
   my $best = -1;

   my $line;
   my @tokens;

   while ( $line = <LOGFILE> ) {
      chomp($line);
      $line =~ s/^\s+//;    #remove leading spaces
      @tokens = split( /\s+/, $line );
      if ( $tokens[0] eq $testname && $tokens[1] eq $attrname ) {
         if ( $tokens[7] > $best ) {
            $best = $tokens[7];
         }
      }
   }
   close(LOGFILE);
   return checkError($best);
}

# Subroutine: findanymax(fileName)
# Purpose: Parses resultDB output to find maximum value for any test
# fileName -- name of log file to open
sub findanymax {

   my $filename = $_[0];
   my $pattern = $_[1];
   open( LOGFILE, $filename );
   my $best = -1;

   my $line;
   my @tokens;
   my $header_found = 0;

   while ( $line = <LOGFILE> ) {
      chomp($line);
      $line =~ s/^\s+//;    #remove leading spaces
      @tokens = split( /\s+/, $line );
      if ( $header_found eq 1 && $tokens[0] =~ /$pattern/ ) {
         if ( $tokens[7] > $best ) {
            $best = $tokens[7];
         }
      }
      if ( $tokens[0] eq "test" ) {
         $header_found = 1;
      }
   }
   close(LOGFILE);
   return checkError($best);
}

# Subroutine: findmean(fileName, testName)
# Purpose: Parses resultDB output to find mean value for a specified test
# fileName -- name of log file to open
# testName -- name of test to look for
sub findmean {

   my $filename = $_[0];
   my $testname = $_[1];

   open( LOGFILE, $filename );
   my $best = -1;
   my $line;
   my @tokens;

   while ( $line = <LOGFILE> ) {
      chomp($line);
      $line =~ s/^\s+//;    #remove leading spaces
      @tokens = split( /\s+/, $line );
      if ( $tokens[0] eq $testname ) {
         if ( $tokens[4] > $best ) {
            $best = $tokens[4];
         }
      }
   }
   close(LOGFILE);
   return checkError($best);
}

sub findanymean {

   my $filename = $_[0];
   my $pattern = $_[1];
   open( LOGFILE, $filename );
   my $best = -1;

   my $line;
   my @tokens;
   my $header_found = 0;

   while ( $line = <LOGFILE> ) {
      chomp($line);
      $line =~ s/^\s+//;    #remove leading spaces
      @tokens = split( /\s+/, $line );
      if ( $header_found eq 1 && $tokens[0] =~ /$pattern/ ) {
         if ( $tokens[4] > $best ) {
            $best = $tokens[7];
         }
      }
      if ( $tokens[0] eq "test" ) {
         $header_found = 1;
      }
   }
   close(LOGFILE);
   return checkError($best);
}
# Subroutine: max(a, b)
# Purpose: returns maximum value
sub max {
   if ( $_[0] > $_[1] ) {
      return $_[0];
   }
   else {
      return $_[1];
   }
}

# Subroutine: isInt(a)
# Purpose: Simple test if a scalar is an integer
sub isInt() {
   return defined $_[0] && $_[0] =~ /^[+-]?\d+$/;
}

# Subroutine: usage
# Purpose: Print command arguments and die
sub usage() {
   print "Usage: perl driver.pl [options]\n";
   print "Mandatory Options\n";
   print "-s      - Problem size (see SHOC wiki for specifics)\n";
   print "-cuda   - Use the cuda version of benchmarks\n";
   print "-opencl - Use the opencl version of the benchmarks\n";
   print "Note -cuda and -opencl are mutually exlcusive.\n\n";

   print "Other options\n";
   print "-n       - Number of nodes to run on\n";
   print "-d       - Number of devices per node\n";
   print "-h       - specify hostfile for parallel runs\n";
   print "-help    - print this message\n";
   print "-bin-dir - location of SHOC bin directory.\n\n";
   print "Note: The driver script assumes it is running from the tools\n";
   print "directory.  Use -bin-dir when you need to run from somwhere else\n\n";

   print "Examples\n";
   print "Test each device in the system serially, using cuda and a large \n";
   print "problem:\n";
   print "\$ perl driver.pl -cuda -s 4\n\n";
   print "Test a cluster with 4 nodes, 3 devs per node, using cuda and a hostfile\n";
   print "\$ perl driver.pl -n 4 -d 3 -cuda -h hostfile_name\n";

   die "Invalid Arguments\n";
}

# Subroutine: checkError
# Purpose: Check to see if a benchmark returned an error
sub checkError() {
   my $ans = $_[0];
   if ( $ans == 0 || $ans == -1 || $ans =~ /inf/ || $ans == 10000000 ) {
      return "BenchmarkError";
   }
   # result DB reports FLT_MAX (guaranteed to be >= 1E+37?) for tests not run
   elsif ($ans >= 1E+37) {
       return "NoResult";
   }
   else {
      return $ans;
   }
}
