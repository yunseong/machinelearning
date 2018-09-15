using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;

namespace Microsoft.ML.Console
{
    /// <summary>
    /// Task Service implemented by means of a thread pool of fixed size.
    /// </summary>
    internal class FixedThreadPool : ThreadPool
    {
        /// <summary>
        /// The maximum concurrency level allowed by this scheduler.
        /// </summary>
        protected readonly int NumThreads;

        protected readonly IntPtr Affinity;

        protected readonly Thread[] Threads;

        [DllImport("kernel32.dll")]
        private static extern int GetCurrentThreadId();

        internal FixedThreadPool(int numThreads, long affinity, Action compute = null)
        {
            if (numThreads <= 0)
            {
                throw new Exception("parallelism " + numThreads + " is less than or equal to 0");
            }

            NumThreads = numThreads;
            Affinity = (IntPtr)affinity;
            Threads = new Thread[numThreads];

            if (compute != null)
            {
                for (int i = 0; i < NumThreads; i++)
                {
                    var thread = new Thread(() =>
                    {
                        PinThread(Affinity);
                        compute();
                    })
                    {
                        IsBackground = true
                    };
                    Threads[i] = thread;
                    thread.Start();
                }
            }
        }
    }
}
