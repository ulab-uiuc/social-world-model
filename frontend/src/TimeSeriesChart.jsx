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

  const displayData = showEstimated && !showBoth ? estimatedData : data;

  const keys = displayData.length > 0
    ? Object.keys(displayData[0]).filter(key => key !== "timestamp")
    : [];

  const height = miniChart ? 150 : 400;
  const margin = miniChart
    ? { top: 5, right: 10, left: 0, bottom: 5 }
    : { top: 10, right: 30, left: 0, bottom: 0 };


  if (displayData.length === 0) {
    return <div>No data available to display</div>;
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
                strokeWidth={miniChart ? 1.5 : 2}
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
