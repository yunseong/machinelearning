using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading;

namespace Microsoft.ML.Console
{
    /// <summary>
    /// Task Service implemented by means of a thread pool of fixed size.
    /// </summary>
    internal class FixedThreadPoolExecutor : FixedThreadPool, IExecutor
    {
        /// <summary>
        /// The list of tasks to be executed.
        /// </summary>
        private readonly ConcurrentQueue<Action> _lowPriorityTasks;

        /// <summary>
        /// The list of tasks to be executed.
        /// </summary>
        private readonly ConcurrentQueue<Action> _highPriorityTasks;

        //private readonly ManualResetEventSlim _lock = new ManualResetEventSlim(false);

        internal FixedThreadPoolExecutor(int numThreads, long affinity, ConcurrentQueue<Action> lowPriorityTasks, ConcurrentQueue<Action> highPriorityTasks) : base(numThreads, affinity, null)
        {
            _lowPriorityTasks = lowPriorityTasks;
            _highPriorityTasks = highPriorityTasks;
            //_lock = new ManualResetEventSlim(false);

            for (int i = 0; i < NumThreads; i++)
            {
                var thread = new Thread(() =>
                {
                    PinThread(Affinity);
                    Compute();
                })
                {
                    IsBackground = true
                };
                Threads[i] = thread;
                thread.Start();
            }
        }

        internal FixedThreadPoolExecutor(int numThreads, long affinity) : this(numThreads, affinity, new ConcurrentQueue<Action>(), new ConcurrentQueue<Action>())
        {
        }

        public void Submit(Action action, bool highPriority = false)
        {
            if (ShuttingDown)
            {
                throw new InvalidOperationException("Shutting down");
            }

            if (highPriority)
            {
                _highPriorityTasks.Enqueue(action);
            }
            else
            {
                _lowPriorityTasks.Enqueue(action);
            }
            //_lock.Set();
        }

        [DllImport("kernel32.dll")]
        private static extern int GetCurrentThreadId();

        private void Compute()
        {
            Action item;
            var osThreadId = GetCurrentThreadId();

            try
            {
                while (true)
                {
                    //_lock.Wait();

                    while (_highPriorityTasks.TryDequeue(out item) || _lowPriorityTasks.TryDequeue(out item))
                    {
                        //_lock.Reset();
                        item.Invoke();
                    }
                }
            }
            catch (Exception ex)
            {
                System.Console.WriteLine("Exception during task execution: " + ex.Message + ex.StackTrace);
                throw ex;
            }
        }
    }
}
