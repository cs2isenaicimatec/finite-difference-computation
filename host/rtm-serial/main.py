
import Profiling.Linux.Perf (readPerfData)
import Profiling.Linux.Perf.Types (PerfData (..))
import System.Environment (getArgs)

main :: IO ()
main = do
   args <- getArgs
   case args of
      [] -> return ()
      (file:_) -> do
         perfData <- readPerfData file
         print $ length $ perfData_events perfData