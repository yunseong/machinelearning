using System;
namespace Microsoft.ML.Console
{
    /// <summary>
    /// The service used to run a Stage on some resource.
    /// </summary>
    public interface IExecutor
    {
        /// <summary>
        /// Submit the action to be scheduled for execution by the service.
        /// </summary>
        void Submit(Action action, bool highPriority = false);
    }
}
