using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using cAlgo.API;
using cAlgo.API.Collections;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;

namespace cAlgo.Robots
{
    public enum TradeDirection
    {
        LongOnly,
        ShortOnly,
        LongShort
    }


    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class TemplateBot : Robot
    {
        [Parameter(DefaultValue = $strategy_name)]
        public string BotName { get; set; }
        
        // Strategy Settings
        public TradeDirection Direction; // LongOnly, ShortOnly, LongShort
        public bool ExitEncodedEntry;
        public int? ExitAfterNBars;
        public int? ExitAfterNDays;
        public bool ExitEndOfWeek;
        public bool ExitEndOfMonth;
        public double? ExitWhenPnLLessThan;
        public double? SLPips;
        public double? TPPips;
        public bool EnableTSL;
        public double TradeSize; // Could be a an Int (Units) or Double (Fraction of Liquidity) but must be converted to double volume
        
        // Private Properties
        private int _barCount; // Used for ExitAfterNBars
        private bool _startCounting; // Used for ExitAfterNBars
        
        private bool _useFixedFractional; // Fixed Fractional
        private double _tradeSize; // Fixed Fractional
        
        // DataSeries
        private DataSeries Open;
        private DataSeries High;
        private DataSeries Low;
        private DataSeries Close;
        private DataSeries Volume;
        private TimeSeries Date;
        
        protected override void OnStart()
        {   
            // Initialize DataSeries
            Open = Bars.OpenPrices;
            High = Bars.HighPrices;
            Low = Bars.LowPrices;
            Close = Bars.ClosePrices;
            Volume = Bars.TickVolumes;
            Date = Bars.OpenTimes;
            
            // To Be Generated in Evoquant. These are just some examples.
            // Instantiate Strategy Settings.
            BotName = BotName;
            Direction = $direction;
            ExitEncodedEntry = $exit_encoded_entry;
            ExitAfterNBars = $exit_after_n_bars;
            ExitAfterNDays = $exit_after_n_days;
            ExitEndOfWeek = $exit_end_of_week;
            ExitEndOfMonth = $exit_end_of_month;
            ExitWhenPnLLessThan = $exit_when_pnl_lessthan;
            //SLPips = null; //TODO: SLPips & TPPips has to be a valid double and Pip units. Percentage and Points must be converted.
            //TPPips = <>; //TODO: SLPipqs & TPPips has to be a valid double and Pip units. Percentage and Points must be converted.
            EnableTSL = false; //TODO: Implement the TSL in Backtesting.py
            TradeSize = $trade_size; //TODO: TradeSize need to be given as Valid Double and converted to volume/units (not lots).
            
            //Debug.Assert(ExitAfterNBars is int? && ExitAfterNBars >= 1, "ExitAfterNBars must be a positive integer");
            
            // Fix TradeSize by Volume/Units for Validity from Template
            if(TradeSize < Symbol.VolumeInUnitsMin && TradeSize >= 1) 
            {
                TradeSize = Symbol.VolumeInUnitsMin;
                _tradeSize = Symbol.NormalizeVolumeInUnits(_tradeSize, RoundingMode.Down);
            }
            else
            {
                _tradeSize = Symbol.NormalizeVolumeInUnits(_tradeSize, RoundingMode.Down);
            }
            
            if (TradeSize > 0 && TradeSize < 1)
            {
                _useFixedFractional = true;
            }
            else 
            {
                _useFixedFractional = false;
            }
            
            Bars.BarOpened +=OnOpen;
            Positions.Opened += OnPositionsOpened;
        }
        
        // Open Bar Event-handler
        private void OnOpen(BarOpenedEventArgs obj) 
        {   
            // Check and Update Bar Counting
            if (_startCounting) // True when there is open position. 
                _barCount++; // Update Number of Bar Count. 
            
            // Run Exits
            RunExitAfterNBars();
            RunExitAfterNDays(obj);
            RunExitWhenPnLLessThan();
            
            // Check and Update Fixed Fractional Size
            if (_useFixedFractional)
            {
                double currentSymbolPrice = obj.Bars.OpenPrices.LastValue;
                Asset SymbolQuoteCurrency = Symbol.QuoteAsset;
                _tradeSize = TradeSize * Account.Balance; // Trade Value based on current state of Balance (Account's Currency)
                _tradeSize = Account.Asset.Convert(SymbolQuoteCurrency, _tradeSize); // In Symbol's Quote Currency
                _tradeSize = Math.Floor(_tradeSize/currentSymbolPrice); // Trade Size in Volume
                _tradeSize = Symbol.NormalizeVolumeInUnits(_tradeSize, RoundingMode.Down); // Normalize to tradable amount in volume
            }
            
            // Check and Update Fixed Stop-Loss (Percentage, Pips, Points)
            SLPips = $stop_loss;
			TPPips = $stop_loss;
            
            // Run Entries
            RunEntries();
        }
        
        protected override void OnTick()
        {
            // Handle price updates here
        }
        
        protected override void OnStop()
        {
            // Handle cBot stop here
        }
        //====================================Testing====================================
        
        //===============================================================================
        private void RunEntries()
        {   
            // Run Entry Code Block
            $indicator_signal_code
            
            var positions = Positions.FindAll(BotName, Symbol.Name); // Get all open positions
            var countPositions = Positions.Count; // Count how many open positions
            
            // Entry
            switch (Direction)
            {
                case TradeDirection.LongOnly:
                if (countPositions > 0) // Return if we have existing long position.
                {
                    Debug.Assert(countPositions == 1, "There is more than 1 position.");
                    // If the root signal is false, check if ExitEncodedEntry is enabled or disabled
                    if(ExitEncodedEntry & !root_signal) // ExitEncodedEntry is Enabled
                    {
                        // Close the position
                        positions[0].Close();
                        break;
                    }else
                    {
                        break;
                    }
                }
                
                Debug.Assert(countPositions == 0, "There is at least 1 position and didn't exit the method");
                if (root_signal) // True for Long Signal
                {
                    ExecuteMarketOrder(TradeType.Buy, Symbol.Name, _tradeSize, BotName, SLPips, TPPips, "", EnableTSL);
                }
                break;
                //----------
                case TradeDirection.ShortOnly:
                if (countPositions > 0)
                {
                    Debug.Assert(countPositions == 1, "There is more than 1 position.");
                    // If the root signal is false, check if ExitEncodedEntry is enabled or disabled
                    if(ExitEncodedEntry & !root_signal) // ExitEncodedEntry is Enabled and root_signal is false
                    {
                        // Close the position
                        positions[0].Close();
                        break;
                    }else
                    {
                        break;
                    }
                }
                
                Debug.Assert(countPositions == 0, "There is at least 1 position and didn't exit the method");
                if (root_signal) // True for Short Signal
                {
                    ExecuteMarketOrder(TradeType.Sell, Symbol.Name, _tradeSize, BotName, SLPips, TPPips, "", EnableTSL);
                }
                
                break;
                //----------
                case TradeDirection.LongShort:
                // We can only have 1 trade at a time. When one direction signal is triggered, the other has to be closed by default.
                if (countPositions > 0) // If there is an existing position (long or short)
                {
                    Debug.Assert(countPositions == 1, "There are more than 1 position");
                    // See if the open position is a long or short.
                    if(positions[0].TradeType == TradeType.Buy) // Current open position is Long
                    {
                        // Check the signals
                        if(root_signal) // True for Long
                        {
                            // Return the method because the signal is long and we have open long position.
                            break;
                        }else // False for Short
                        {
                            if(ExitEncodedEntry) // Check if ExitEncodedEntry is enabled
                            {
                                // Close the long position and place a short market order
                                positions[0].Close();
                            }else
                            {
                                break; // Break if ExitEncodedEntry not enabled
                            }
                        }
                    }else // Current open position is Short
                    {
                        // Check the signals
                        if (root_signal) // True for Long
                        {
                            if(ExitEncodedEntry) // Check if ExitEncodedEntry is enabled
                            {
                                // Close the short position and place a long market order
                                positions[0].Close();
                            }else
                            {
                                break; // Break if ExitEncodedEntry not enabled
                            }
                        }else // False for Short
                        {
                            // Return the method because the signal is short and we have open short position.
                            break;
                        }
                    }
                }
                
                Debug.Assert(countPositions == 0, "There is at least 1 position and didn't exit the method");
                // If there are no open position, see which signal is triggered. True for Long and False for Short.
                // We always check the signal.
                if (root_signal) // True for Long
                {
                    ExecuteMarketOrder(TradeType.Buy, Symbol.Name, _tradeSize, BotName, SLPips, TPPips, "", EnableTSL);
                }else // False for Short
                {
                    ExecuteMarketOrder(TradeType.Sell, Symbol.Name, _tradeSize, BotName, SLPips, TPPips, "", EnableTSL);
                }
                break;
            }
        }
        //====================================Exit Methods====================================
        private void RunExitAfterNBars()
        {
            if(ExitAfterNBars == null) {return;} // Return the method if ExitAfterNBars is disabled
            // We Assert that ExitAfterNBars is a valid positive integer
            
            var positions = Positions.FindAll(BotName, Symbol.Name); // Get all open positions
            var countPositions = Positions.Count; // Count how many open positions
            
            if (countPositions == 0) {return;} // No Position. There is nothing to exit.
            
            Debug.Assert(countPositions == 1, "There should be only 1 position at a time");
            
            Position position = positions[0];
            
            if (_barCount >= ExitAfterNBars)
            {
                position.Close();
            }
        }
        
        private void RunExitAfterNDays(BarOpenedEventArgs obj)
        {
            if(ExitAfterNDays == null) {return;} // Return the method if ExitAfterNDays is disabled
            // We Assert that ExitAfterNDays is a valid positive integer
            
            var positions = Positions.FindAll(BotName, Symbol.Name); // Get all open positions
            var countPositions = Positions.Count; // Count how many open positions
            
            if (countPositions == 0) {return;} // No Position. There is nothing to exit.
            
            Debug.Assert(countPositions == 1, "There should be only 1 position at a time");
            
            Position position = positions[0];
            
            var NowDatetime = obj.Bars.OpenTimes.LastValue;
            var DateDiff = NowDatetime.Subtract(position.EntryTime).Days;
            if (DateDiff >= ExitAfterNDays)
            {
                position.Close();
            }
        }
        
        private void RunExitWhenPnLLessThan()
        {
            /*Ideally, this should be in OnTick Event-Handler*/
            var positions = Positions.FindAll(BotName, Symbol.Name); // Get all open positions
            var countPositions = Positions.Count; // Count how many open positions
            
            if (ExitWhenPnLLessThan == null) {return;} //Return method if ExitWhenPnLLessThan is disabled
            if (countPositions == 0) {return;} //Return method when there are no open positions to close
            
            Debug.Assert(countPositions == 1, "There should be only 1 position at a time");
            Position position = positions[0];
            
            // Close the position if the net profit is less than a given fixed dollar amount threshold (must be negative).
            if (position.NetProfit <= ExitWhenPnLLessThan) {position.Close();}
        }
        
        private void RunExitEndOfWeek()
        {
            // To be implemented in the future
        }
        private void RunExitEndOfMonth()
        {
            // To be implemented in the future
        }
        //====================================Other Event Handlers====================================
        void OnPositionsOpened(PositionOpenedEventArgs obj)
        {
            _startCounting = true; // Start Counting Number of bars after a position is opened.
            _barCount = 0; // Set intially to 0.
            // Remind that when a position is opened, its on the Open Price of the Bar.
        }
        
        void OnPositionsClosed(PositionClosedEventArgs obj)
        {
            // Reset Number of Bars Counting.
            _startCounting = false;
        }
        
        //====================================Signal Methods====================================
        private bool SeriesCrossAboveSeries(DataSeries Series1, DataSeries Series2)
        {
            bool result = Series1.Last(1+0) > Series2.Last(1+0) & Series1.Last(1+1) < Series2.Last(1+1);
            return result;
        }
        private bool SeriesCrossBelowSeries(DataSeries Series1, DataSeries Series2)
        {
            bool result = Series1.Last(1+0) < Series2.Last(1+0) & Series1.Last(1+1) > Series2.Last(1+1);
            return result;
        }
        
        private bool SeriesIsAboveSeries(DataSeries Series1, DataSeries Series2)
        {
            bool result = Series1.Last(1+0) > Series2.Last(1+0);
            return result;
        }
        private bool SeriesIsBelowSeries(DataSeries Series1, DataSeries Series2)
        {
            bool result = Series1.Last(1+0) < Series2.Last(1+0);
            return result;
        }
        
        private bool IsIncr(DataSeries Series, int Period)
        {   
            List<Boolean> bools = new List<Boolean>();
            for (int i = 0; i < Period; i++)
            {
                bools.Add(Series.Last(1+i) > Series.Last(1+i+1));
            }
            bool AllTrue = bools.All(x => x);
            if(AllTrue) {return true;}
            else {return false;}
        }
        private bool IsDecr(DataSeries Series, int Period)
        {   
            List<Boolean> bools = new List<Boolean>();
            for (int i = 0; i < Period; i++)
            {
                bools.Add(Series.Last(1+i) < Series.Last(1+i+1));
            }
            bool AllTrue = bools.All(x => x);
            if(AllTrue) {return true;}
            else {return false;}
        }
        
        private bool IsHighest(DataSeries Series, int Period)
        {
            var CurrentValue = Series.Last(1+0); // Assume that Current is the Max
            for (int i = 1; i < Period; i++)
            {   
                var PrevVal = Series.Last(1+i);
                if(PrevVal > CurrentValue) // We found a past value higher than the current
                {
                    return false;
                }
            }
            return true;
        }
        private bool IsLowest(DataSeries Series, int Period)
        {
            var CurrentValue = Series.Last(1+0); // Assume that Current is the Min
            for (int i = 1; i < Period; i++)
            {   
                var PrevVal = Series.Last(1+i);
                if(PrevVal < CurrentValue) // We found a past value higher than the current
                {
                    return false;
                }
            }
            return true;
        }
        //====================================Utils====================================
        // Convertion Methods
    }
}