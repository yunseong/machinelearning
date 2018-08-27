rm -r -Force ~/Source/nuget_repo/*
cd C:\Users\b-yule.REDMOND\Source\Repos\mldotnet2
rm -r -Force bin/packages
./build.cmd -BuildPackages
c:\nuget\nuget.exe add C:\Users\b-yule.REDMOND\Source\Repos\mldotnet2\bin/packages/Microsoft.ML.CpuMath.0.5.0-preview-26819-0.nupkg -source C:\Users\b-yule.REDMOND\Source\nuget_repo
c:\nuget\nuget.exe add C:\Users\b-yule.REDMOND\Source\Repos\mldotnet2\bin/packages/Microsoft.ML.CpuMath.symbols.0.5.0-preview-26819-0.nupkg -source C:\Users\b-yule.REDMOND\Source\nuget_repo
c:\nuget\nuget.exe add C:\Users\b-yule.REDMOND\Source\Repos\mldotnet2\bin/packages/Microsoft.ML.HalLearners.0.5.0-preview-26819-0.nupkg -source C:\Users\b-yule.REDMOND\Source\nuget_repo
c:\nuget\nuget.exe add C:\Users\b-yule.REDMOND\Source\Repos\mldotnet2\bin/packages/Microsoft.ML.HalLearners.symbols.0.5.0-preview-26819-0.nupkg -source C:\Users\b-yule.REDMOND\Source\nuget_repo
c:\nuget\nuget.exe add C:\Users\b-yule.REDMOND\Source\Repos\mldotnet2\bin/packages/Microsoft.ML.ImageAnalytics.0.5.0-preview-26819-0.nupkg -source C:\Users\b-yule.REDMOND\Source\nuget_repo
c:\nuget\nuget.exe add C:\Users\b-yule.REDMOND\Source\Repos\mldotnet2\bin/packages/Microsoft.ML.ImageAnalytics.symbols.0.5.0-preview-26819-0.nupkg -source C:\Users\b-yule.REDMOND\Source\nuget_repo
c:\nuget\nuget.exe add C:\Users\b-yule.REDMOND\Source\Repos\mldotnet2\bin/packages/Microsoft.ML.LightGBM.0.5.0-preview-26819-0.nupkg -source C:\Users\b-yule.REDMOND\Source\nuget_repo
c:\nuget\nuget.exe add C:\Users\b-yule.REDMOND\Source\Repos\mldotnet2\bin/packages/Microsoft.ML.LightGBM.symbols.0.5.0-preview-26819-0.nupkg -source C:\Users\b-yule.REDMOND\Source\nuget_repo
c:\nuget\nuget.exe add C:\Users\b-yule.REDMOND\Source\Repos\mldotnet2\bin/packages/Microsoft.ML.Onnx.0.5.0-preview-26819-0.nupkg -source C:\Users\b-yule.REDMOND\Source\nuget_repo
c:\nuget\nuget.exe add C:\Users\b-yule.REDMOND\Source\Repos\mldotnet2\bin/packages/Microsoft.ML.Onnx.symbols.0.5.0-preview-26819-0.nupkg -source C:\Users\b-yule.REDMOND\Source\nuget_repo
c:\nuget\nuget.exe add C:\Users\b-yule.REDMOND\Source\Repos\mldotnet2\bin/packages/Microsoft.ML.Parquet.0.5.0-preview-26819-0.nupkg -source C:\Users\b-yule.REDMOND\Source\nuget_repo
c:\nuget\nuget.exe add C:\Users\b-yule.REDMOND\Source\Repos\mldotnet2\bin/packages/Microsoft.ML.Parquet.symbols.0.5.0-preview-26819-0.nupkg -source C:\Users\b-yule.REDMOND\Source\nuget_repo
c:\nuget\nuget.exe add C:\Users\b-yule.REDMOND\Source\Repos\mldotnet2\bin/packages/Microsoft.ML.0.5.0-preview-26819-0.nupkg -source C:\Users\b-yule.REDMOND\Source\nuget_repo
c:\nuget\nuget.exe add C:\Users\b-yule.REDMOND\Source\Repos\mldotnet2\bin/packages/Microsoft.ML.symbols.0.5.0-preview-26819-0.nupkg -source C:\Users\b-yule.REDMOND\Source\nuget_repo
