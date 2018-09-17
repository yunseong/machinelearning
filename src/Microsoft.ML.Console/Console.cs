// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime;
using System.Linq;
using System.Collections.Generic;
using static Microsoft.ML.Runtime.Tools.Console.AttendeeModel;
using Microsoft.ML.Runtime.Api;
using System.Diagnostics;
using System.IO;
using static Microsoft.ML.Runtime.Tools.Console.AmazonModel;
using System.Threading;
using Microsoft.ML.Console;

namespace Microsoft.ML.Runtime.Tools.Console
{
    public static class Console
    {
        private static string _metadataFile = @"%USERPROFILE%\Source\Repos\TLC\DataCollector\model-data-file-paths.txt";
        private static string _amznInputRecordFile = @"%USERPROFILE%\OneDrive - Microsoft\models\AmazonReview_1024records.csv";
        private static string _atndInputRecordFile = @"%USERPROFILE%\OneDrive - Microsoft\models\AttendeeCountPrediction_1024records.csv";

        private static int _numExecutors = 1;
        private static int _numIterations = 1;
        private static int _numRequestHandler = 1;
        private static int _numStageHandlerThreads = 1;

        private static void PrintHelp()
        {
            System.Console.WriteLine("PARAMETERS:");
            System.Console.WriteLine("\t-nex <number of executors>; default is {0}>", _numExecutors);
            System.Console.WriteLine("\t-i <number of iterations; default is {0}>", _numIterations);
            System.Console.WriteLine("\t-nsht <number of stage handler threads>; default is {0}>", _numStageHandlerThreads);
            System.Console.WriteLine("\t-nbrt <number of request threads>; default is {0}>", _numRequestHandler);

        }

        private static void ParseArgs(string[] args)
        {
            for (int j = 0; j < args.Count(); j++)
            {
                switch (args[j])
                {
                    case "-nex":
                        _numExecutors = int.Parse(args[j + 1]);
                        j++;
                        break;
                    case "-nbrt":
                        _numRequestHandler = int.Parse(args[j + 1]);
                        j++;
                        break;
                    case "-nsht":
                        _numStageHandlerThreads = int.Parse(args[j + 1]);
                        j++;
                        break;
                    case "-i":
                        _numIterations = int.Parse(args[j + 1]);
                        j++;
                        break;
                    case "-h":
                        PrintHelp();
                        Environment.Exit(-1);
                        break;
                    default:
                        System.Console.WriteLine("unknown option: " + args[j]);
                        Environment.Exit(-1);
                        break;
                }
            }
        }

        public static int Main(string[] args)
        { //=> Maml.Main(args);
            IHostEnvironment env = new TlcEnvironment();
            //var atndBatches = AttendeeModel.Parse(Environment.ExpandEnvironmentVariables(_atndInputRecordFile), batchSize);
            //var amznBatches = AmazonModel.Parse(Environment.ExpandEnvironmentVariables(_amznInputRecordFile), batchSize);

            ParseArgs(args);

            LoadModels(env,
                out IDictionary<string, BatchPredictionEngine<AttendeeData, AttendeeResult>> atndEngines,
                out IDictionary<string, BatchPredictionEngine<AmazonData, AmazonResult>> amznEngines);

            var predictionStartTime = Stopwatch.GetTimestamp();
            //RunLatency(atndEngines, atndBatches, amznEngines, amznBatches);
            //RunThroughput(atndEngines, atndBatches, amznEngines, amznBatches);

            var predictionEndTime = Stopwatch.GetTimestamp();
            System.Console.WriteLine("Done. Total prediction Time: {0}",
                (predictionEndTime - predictionStartTime) * 1000.0 / Stopwatch.Frequency);
            return 0;
        }

        private static double RunInParallel<TData, TResult>(
            ThreadLocal<BatchPredictionEngine<TData, TResult>> engine,
            List<TData[]> batches,
            IExecutor[] executors,
            int numThreadsPerExecutor,
            int numRequestors,
            int numRepeat)
            where TData : class, new()
            where TResult : class, new()
        {
            CountdownEvent warmupCde = new CountdownEvent(executors.Length * numThreadsPerExecutor);
            foreach (var executor in executors)
            {
                for (int i = 0; i < numThreadsPerExecutor; i++)
                {
                    // Does it make sure we cache?
                    executor.Submit(() =>
                    {
                        var result = engine.Value.Predict(batches[0], false);
                        var resultStr = string.Join(',', result.Select(x =>
                        {
                            if (x is AttendeeResult)
                            {
                                return (x as AttendeeResult).Score;
                            }
                            else if (x is AmazonResult)
                            {
                                return (x as AmazonResult).Score;
                            }
                            else
                            {
                                throw new Exception("Unsupported result type");
                            }
                        }));
                        warmupCde.Signal();
                    });

                }
            }
            warmupCde.Wait();

            CountdownEvent outerCde = new CountdownEvent(numRepeat * batches.Count);
            CountdownEvent cde = new CountdownEvent(1);

            Action compute = () =>
            {
                cde.Wait();

                for (int bIdx = 0; bIdx < numRepeat * batches.Count / numRequestors; bIdx++)
                {
                    // Internally an executor is chosen in round-robin.
                    executors[bIdx % executors.Length].Submit(() =>
                    {
                        //results.Enqueue(engine.Predict(batches[bIdx % batches.Count]));
                        var result = engine.Value.Predict(batches[bIdx % batches.Count], true);
                        var resultStr = string.Join(',', result.Select(x =>
                        {
                            if (x is AttendeeResult)
                            {
                                return (x as AttendeeResult).Score;
                            }
                            else if (x is AmazonResult)
                            {
                                return (x as AmazonResult).Score;
                            }
                            else
                            {
                                throw new Exception("Unsupported result type");
                            }
                        }));
                        outerCde.Signal();
                    });
                }
            };

            int cores = RegisterRequestorThreads(compute, numRequestors);

            var startTime = Stopwatch.GetTimestamp();
            cde.Signal();

            outerCde.Wait();
            var elapsedTime = (Stopwatch.GetTimestamp() - startTime) * 1000.0 / Stopwatch.Frequency;
            System.Console.WriteLine("Prediction took {0} ms", elapsedTime);

            return elapsedTime;
        }

        private static void RunThroughput(
            IDictionary<string, BatchPredictionEngine<AttendeeData, AttendeeResult>> atndEngines, List<AttendeeData[]> atndBatches,
            IDictionary<string, BatchPredictionEngine<AmazonData, AmazonResult>> amznEngines, List<AmazonData[]> amznBatches,
            int numExecutors,
            int numRequestors,
            int numRepeat,
            int numThreadsPerExecutor = 1)
        {
            IExecutor[] executors = new IExecutor[numExecutors];
            for (int i = 0; i < numExecutors; i++)
            {
                executors[i] = new FixedThreadPoolExecutor(numThreadsPerExecutor, 4 << i);
            }

            var totalPredictionTime = 0.0;
            /*
            foreach (var engine in atndEngines)
            {
                totalPredictionTime += RunInParallel(engine.Value, atndBatches, executors, numThreadsPerExecutor, numRequestors, numRepeat);
            }
            foreach (var engine in amznEngines)
            {
                totalPredictionTime += RunInParallel(engine.Value, amznBatches, executors, numThreadsPerExecutor, numRequestors, numRepeat);
            }
            */
            System.Console.WriteLine("Total time is: " + totalPredictionTime);
        }

        private static void RunLatency(IDictionary<string, BatchPredictionEngine<AttendeeData, AttendeeResult>> atndEngines, List<AttendeeData[]> atndBatches, IDictionary<string, BatchPredictionEngine<AmazonData, AmazonResult>> amznEngines, List<AmazonData[]> amznBatches)
        {
            foreach (var engine in atndEngines)
            {
                foreach (var batch in atndBatches)
                {
                    var startTime = Stopwatch.GetTimestamp();
                    var result = engine.Value.Predict(batch, false);
                    var resultStr = string.Join(',', result.Select(x => x.Score));
                    var latency = (Stopwatch.GetTimestamp() - startTime) * 1000.0 / Stopwatch.Frequency;
                    System.Console.WriteLine("File: {0} Latency: {1} Result: {2}", engine.Key, latency, resultStr);
                }
            }
            foreach (var engine in amznEngines)
            {
                foreach (var batch in amznBatches)
                {
                    var startTime = Stopwatch.GetTimestamp();
                    var result = engine.Value.Predict(batch, false);
                    var resultStr = string.Join(',', result.Select(x => x.Score));
                    var latency = (Stopwatch.GetTimestamp() - startTime) * 1000.0 / Stopwatch.Frequency;
                    System.Console.WriteLine("File: {0} Latency: {1} Result: {2}", engine.Key, latency, resultStr);
                }
            }
        }

        private static void RunLatencySingle(string modelFile,
            BatchPredictionEngine<AttendeeData, AttendeeResult> atndEngines,
            List<AttendeeData[]> atndBatches,
            BatchPredictionEngine<AmazonData, AmazonResult> amznEngines,
            List<AmazonData[]> amznBatches,
            bool print = true)
        {
            if (atndEngines != null)
            {
                foreach (var batch in atndBatches)
                {
                    var startTime = Stopwatch.GetTimestamp();
                    var result = atndEngines.Predict(batch, false);
                    var resultStr = string.Join(',', result.Select(x => x.Score));
                    var latency = (Stopwatch.GetTimestamp() - startTime) * 1000.0 / Stopwatch.Frequency;
                    if (print)
                    {
                        System.Console.WriteLine("File: {0} Latency: {1} Result: {2}", modelFile, latency, resultStr);
                    }
                }
            }
            else if (amznEngines != null)
            {

                foreach (var batch in amznBatches)
                {
                    var startTime = Stopwatch.GetTimestamp();
                    var result = amznEngines.Predict(batch, false);
                    var resultStr = string.Join(',', result.Select(x => x.Score));
                    var latency = (Stopwatch.GetTimestamp() - startTime) * 1000.0 / Stopwatch.Frequency;
                    if (print)
                    {
                        System.Console.WriteLine("File: {0} Latency: {1} Result: {2}", modelFile, latency, resultStr);
                    }
                }
            }
        }

        private static void LoadModels(IHostEnvironment env,
            out IDictionary<string, BatchPredictionEngine<AttendeeData, AttendeeResult>> atndEngines,
            out IDictionary<string, BatchPredictionEngine<AmazonData, AmazonResult>> amznEngines)
        {
            int batchSize = 1024;
            var atndBatches = AttendeeModel.Parse(Environment.ExpandEnvironmentVariables(_atndInputRecordFile), batchSize);
            var amznBatches = AmazonModel.Parse(Environment.ExpandEnvironmentVariables(_amznInputRecordFile), batchSize);

            var numModels = 0;
            var totalStartTime = Stopwatch.GetTimestamp();
            atndEngines = new Dictionary<string, BatchPredictionEngine<AttendeeData, AttendeeResult>>();
            amznEngines = new Dictionary<string, BatchPredictionEngine<AmazonData, AmazonResult>>();

            IExecutor[] executors = new IExecutor[_numExecutors];
            for (int i = 0; i < _numExecutors; i++)
            {
                executors[i] = new FixedThreadPoolExecutor(1, 4 << i);
            }

            using (var reader = new StreamReader(Environment.ExpandEnvironmentVariables(_metadataFile)))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    string[] splits = line.Split();
                    var modelPath = Environment.ExpandEnvironmentVariables(splits[0]);

                    var startTime = Stopwatch.GetTimestamp();
                    if (modelPath.Contains("Attendee"))
                    {
                        var engine = new ThreadLocal<BatchPredictionEngine<AttendeeData, AttendeeResult>>(() =>
                        {
                            return AttendeeModel.CreateEngine(env, Environment.ExpandEnvironmentVariables(modelPath));
                        });
                        // Throughput
                        RunInParallel(engine, atndBatches, executors, _numStageHandlerThreads, _numRequestHandler, _numIterations);

                        // Latency
                        //RunLatencySingle(modelPath, engine.Value, atndBatches.Take(10).ToList(), null, amznBatches.Take(10).ToList(), false);
                        //RunLatencySingle(modelPath, engine, atndBatches, null, amznBatches);
                        //atndEngines.Add(modelPath, engine);
                    }
                    else if (modelPath.Contains("Amazon"))
                    {
                        var engine = new ThreadLocal<BatchPredictionEngine<AmazonData, AmazonResult>>(() =>
                        {
                            return AmazonModel.CreateEngine(env, Environment.ExpandEnvironmentVariables(modelPath));
                        });
                        RunInParallel(engine, amznBatches, executors, _numStageHandlerThreads, _numRequestHandler, _numIterations);
                        //                RunLatencySingle(modelPath, null, atndBatches.Take(10).ToList(), engine, amznBatches.Take(10).ToList(), false);
                        //                RunLatencySingle(modelPath, null, atndBatches, engine, amznBatches);
                        //amznEngines.Add(modelPath, engine);
                    }
                    var loadingTime = (Stopwatch.GetTimestamp() - startTime) * 1000.0 / Stopwatch.Frequency;
                    using (Process proc = Process.GetCurrentProcess())
                    {
                        var memory = proc.PrivateMemorySize64;
                        System.Console.WriteLine("Load {0} in {1} ms", modelPath, loadingTime);
                        System.Console.WriteLine("Memory after loading {0} {1}", modelPath, memory);
                    }

                    // TODO: Other type of models

                    numModels++;
                }
            }

            var totalLoadingTime = (Stopwatch.GetTimestamp() - totalStartTime) * 1000.0 / Stopwatch.Frequency;
            System.Console.WriteLine("Load {0} models in {1} ms", numModels, totalLoadingTime);
        }

        private static int RegisterRequestorThreads(Action compute, int numThreads)
        {
            Thread.BeginThreadAffinity();

            for (int i = 0; i < numThreads; i++)
            {
                var aff = new IntPtr(2 << i);
                var thread = new Thread(() =>
                {
                    Microsoft.ML.Console.ThreadPool.PinThread(aff);
                    compute();
                })
                {
                    IsBackground = true
                };
                thread.Start();
            }
            Thread.EndThreadAffinity();

            return numThreads;
        }
    }
}
