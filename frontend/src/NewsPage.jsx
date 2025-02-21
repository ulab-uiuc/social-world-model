import React, { useState, useEffect } from "react";
import axios from "axios";
import "./NewsPage.css";

const API_BASE_URL =
  window.location.hostname === "localhost"
    ? "http://localhost:5001"
    : `http://${window.location.hostname}:8000`;

const API_NEWS_URL = `${API_BASE_URL}/api/news`;

const NewsPage = () => {
  const [news, setNews] = useState([]);

  useEffect(() => {
    const fetchNews = async () => {
      try {
        const response = await axios.get(API_NEWS_URL);
        setNews(response.data);
      } catch (error) {
        console.error("Error fetching news:", error);
      }
    };

    fetchNews();
  }, []);

  return (
    <div className="news-container">
      <h2 className="news-title">Latest News</h2>
      {news.length === 0 ? (
        <p className="no-news">No news available</p>
      ) : (
        <ul className="news-list">
          {news.map((item, index) => (
            <li key={index} className="news-item">
              <a href={item.link} target="_blank" rel="noopener noreferrer">{item.title}</a>
              <span className="news-date"> ({new Date(item.timestamp).toLocaleString()})</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default NewsPage;
