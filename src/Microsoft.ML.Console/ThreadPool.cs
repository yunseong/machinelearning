using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;

namespace Microsoft.ML.Console
{
    /// <summary>
    /// An astraction of thread pool.
    /// </summary>
    internal abstract class ThreadPool
    {
        protected volatile bool ShuttingDown;

        [DllImport("kernel32.dll")]
        private static extern int GetCurrentThreadId();

        public void Shutdown()
        {
            ShuttingDown = true;
        }

        internal static void PinThread(IntPtr affinity)
        {
            var osThreadId = GetCurrentThreadId();
            ProcessThread thread = Process.GetCurrentProcess().Threads.Cast<ProcessThread>().Where(t => t.Id == osThreadId).Single();

            thread.ProcessorAffinity = affinity;
            thread.PriorityLevel = ThreadPriorityLevel.Highest;

            System.Console.WriteLine("Thread Id {0} Affinity Set to {1}.", thread.Id, affinity);
            Thread.BeginThreadAffinity();
        }
    }
}
