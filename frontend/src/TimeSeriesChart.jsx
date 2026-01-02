import React, { useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const TimeSeriesChart = ({
  data = [],
  estimatedData = [],
  showBoth = false,
  showEstimated = false,
  miniChart = false
}) => {
  useEffect(() => {
    console.log("TimeSeriesChart props:", {
      dataLength: data.length,
      estimatedDataLength: estimatedData.length,
      showBoth,
      showEstimated
    });
  }, [data, estimatedData, showBoth, showEstimated]);

  const cleanData = (rawData) => {
    if (!Array.isArray(rawData) || rawData.length === 0) {
      return [];
    }

    // Sort by timestamp first
    const sortedData = [...rawData].sort((a, b) => 
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    const cleanedData = [];
    const historicalValues = {}; // Keep track of recent values for smoothing
    const windowSize = 5; // Larger window for better smoothing

    sortedData.forEach((item, index) => {
      const cleanedItem = { timestamp: item.timestamp };
      
      Object.keys(item).forEach(key => {
        if (key !== 'timestamp') {
          let value = item[key];
          
          // Convert to number and validate
          if (typeof value === 'string') {
            value = parseFloat(value);
          }
          
          // Filter out invalid values (NaN, null, undefined)
          if (isNaN(value) || value === null || value === undefined) {
            // Use last known value if available
            if (historicalValues[key] && historicalValues[key].length > 0) {
              value = historicalValues[key][historicalValues[key].length - 1];
            } else {
              value = 0;
            }
          }
          
          // Initialize historical values array if needed
          if (!historicalValues[key]) {
            historicalValues[key] = [];
          }
          
          // Detect extreme outliers and apply aggressive smoothing
          if (historicalValues[key].length >= 2) {
            const recent = historicalValues[key].slice(-Math.min(3, historicalValues[key].length));
            const avgRecent = recent.reduce((a, b) => a + b, 0) / recent.length;
            const diff = Math.abs(value - avgRecent);
            
            // Much more aggressive outlier detection and smoothing
            const threshold = Math.max(5, avgRecent * 0.15); // Very low threshold for outliers
            if (diff > threshold) {
              console.warn(`Detected outlier for ${key}: ${value}, smoothing with recent average ${avgRecent} (diff: ${diff}, threshold: ${threshold})`);
              
              // Extremely aggressive smoothing for large outliers
              if (diff > Math.max(15, avgRecent * 0.4)) {
                value = avgRecent * 0.95 + value * 0.05; // 95% historical trend
              } else {
                value = avgRecent * 0.9 + value * 0.1; // 90% historical trend  
              }
            }
          }
          
          // Clamp values to reasonable range
          if (value < 0) {
            value = 0;
          } else if (value > 10000) {
            value = Math.min(value, 10000);
          }
          
          // Update historical values (keep only recent window)
          historicalValues[key].push(value);
          if (historicalValues[key].length > windowSize) {
            historicalValues[key].shift();
          }
          
          // Apply exponential moving average for smoother transitions
          if (historicalValues[key].length >= 2) {
            const alpha = 0.25; // Lower alpha = more smoothing
            const prevValue = historicalValues[key][historicalValues[key].length - 1];
            value = alpha * value + (1 - alpha) * prevValue;
          }
          
          // Additional moving average for extra smoothing
          if (historicalValues[key].length >= windowSize) {
            const movingAvg = historicalValues[key].reduce((a, b) => a + b, 0) / historicalValues[key].length;
            value = movingAvg * 0.7 + value * 0.3; // Strong blend with moving average
          }
          
          cleanedItem[key] = Math.round(value * 100) / 100; // Round to 2 decimal places
        }
      });
      
      cleanedData.push(cleanedItem);
    });

    const finalData = cleanedData.filter(item => {
      // Remove entries where all non-timestamp values are 0
      const nonTimestampKeys = Object.keys(item).filter(key => key !== 'timestamp');
      return nonTimestampKeys.some(key => item[key] > 0);
    });

    // Apply final Gaussian smoothing for ultra-smooth curves
    return applyGaussianSmoothing(finalData, 2);
  };

  const applyGaussianSmoothing = (data, radius) => {
    if (data.length <= radius * 2) return data;
    
    const result = [...data];
    const weights = [];
    let weightSum = 0;
    
    // Generate Gaussian weights
    for (let i = -radius; i <= radius; i++) {
      const weight = Math.exp(-(i * i) / (2 * radius * radius));
      weights.push(weight);
      weightSum += weight;
    }
    
    // Normalize weights
    for (let i = 0; i < weights.length; i++) {
      weights[i] /= weightSum;
    }
    
    // Apply smoothing
    for (let i = radius; i < data.length - radius; i++) {
      const smoothedItem = { timestamp: data[i].timestamp };
      
      Object.keys(data[i]).forEach(key => {
        if (key !== 'timestamp') {
          let smoothedValue = 0;
          
          for (let j = -radius; j <= radius; j++) {
            smoothedValue += data[i + j][key] * weights[j + radius];
          }
          
          smoothedItem[key] = Math.round(smoothedValue * 100) / 100;
        }
      });
      
      result[i] = smoothedItem;
    }
    
    return result;
  };

  // Use raw data without any cleaning or smoothing
  const displayData = showEstimated && !showBoth ? estimatedData : data;

  const keys = displayData.length > 0
    ? Object.keys(displayData[0]).filter(key => key !== "timestamp")
    : [];

  const height = miniChart ? 150 : 400;
  const margin = miniChart
    ? { top: 5, right: 10, left: 0, bottom: 5 }
    : { top: 10, right: 30, left: 0, bottom: 0 };


  if (displayData.length === 0) {
    return (
      <div style={{ 
        width: "100%", 
        height: miniChart ? 150 : 400, 
        display: "flex", 
        alignItems: "center", 
        justifyContent: "center",
        color: "#a0a0b8",
        fontSize: "0.875rem"
      }}>
        No data available
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <LineChart
          data={displayData}
          margin={margin}
        >
          {!miniChart && <CartesianGrid strokeDasharray="3 3" />}
          <XAxis
            dataKey="timestamp"
            tick={!miniChart}
            hide={miniChart}
          />
          <YAxis hide={miniChart} />
          {!miniChart && <Tooltip />}
          {!miniChart && <Legend />}

          {}
          {!showEstimated || showBoth ?
            keys.map((key, index) => (
              <Line
                key={`actual-${key}`}
                type="monotone"
                dataKey={key}
                stroke={["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#8e44ad"][index % 5]}
                name={showBoth ? `${key} (Actual)` : key}
                dot={false}
                strokeWidth={miniChart ? 2 : 3}
                connectNulls={true}
                strokeLinecap="round"
                strokeLinejoin="round"
                tension={0.3}
              />
            )) : null
          }

          {}
          {showBoth && estimatedData.length > 0 ?
            keys.map((key, index) => (
              <Line
                key={`estimated-${key}`}
                type="monotone"
                data={estimatedData}
                dataKey={key}
                stroke={["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#8e44ad"][index % 5]}
                name={`${key} (Estimated)`}
                dot={false}
                strokeDasharray="5 5"
                strokeWidth={miniChart ? 1.5 : 2}
              />
            )) : null
          }
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TimeSeriesChart;
