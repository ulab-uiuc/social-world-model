import React, { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import "./NewsPage.css";
import TimeSeriesChart from "./TimeSeriesChart";

const API_BASE_URL =
  window.location.hostname === "localhost"
    ? "http://localhost:5001"
    : `http://${window.location.hostname}:5001`;

const API_NEWS_URL = `/api/news`;

const API_BACKEND_URL = window.location.hostname === "localhost"
  ? "http://localhost:5000"
  : `http://${window.location.hostname}:5000`;

const API_NEWS_CARDS_URL = `${API_BACKEND_URL}/api/news_cards`;
const API_VOTE_HISTORY_URL = `${API_BACKEND_URL}/api/vote_history`;

const NewsPage = () => {
  const [news, setNews] = useState([]);
  const [expandedNewsId, setExpandedNewsId] = useState(null);
  const [relatedCards, setRelatedCards] = useState({});
  const [cardHistory, setCardHistory] = useState({});
  const navigate = useNavigate();

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

  const fetchRelatedCards = async (newsId) => {
    try {
      console.log(`Fetching related cards for news ID: ${newsId}`);

      if (!newsId) {
        console.error("Invalid news ID:", newsId);
        return;
      }

      if (!relatedCards[newsId]) {
        const response = await axios.get(`${API_NEWS_CARDS_URL}/${newsId}`);
        console.log(`API response for news ID ${newsId}:`, response.data);

        setRelatedCards(prev => ({
          ...prev,
          [newsId]: response.data
        }));

        for (const card of response.data) {
          fetchCardHistory(card.card_id);
        }
      }
    } catch (error) {
      console.error(`Error fetching related cards for news ID ${newsId}:`, error);
      console.error("Error details:", error.response || error.message);

      setRelatedCards(prev => ({
        ...prev,
        [newsId]: []
      }));
    }
  };

  const fetchCardHistory = async (cardId) => {
    try {
      if (!cardHistory[cardId]) {
        const response = await axios.get(`${API_VOTE_HISTORY_URL}/${cardId}`);
        const formattedData = response.data.map(entry => ({
          timestamp: entry.timestamp,
          ...entry.votes,
        }));
        setCardHistory(prev => ({
          ...prev,
          [cardId]: formattedData
        }));
      }
    } catch (error) {
      console.error("Error fetching card history:", error);
    }
  };

  const toggleNewsExpand = (newsId) => {
    if (expandedNewsId === newsId) {
      setExpandedNewsId(null);
    } else {
      setExpandedNewsId(newsId);
      fetchRelatedCards(newsId);
    }
  };

  const navigateToCardDetails = (cardId) => {
    navigate(`/details/${cardId}`);
  };

  return (
    <div className="news-container">
      <h2 className="news-title">Latest News</h2>
      {news.length === 0 ? (
        <p className="no-news">No news available</p>
      ) : (
        <ul className="news-list">
          {news.map((item, index) => (
            <li key={index} className="news-item-container">
              <div className="news-item">
                <div className="news-content">
                  <a
                    href={item.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="news-link"
                  >
                    {item.title}
                  </a>
                  <span className="news-date">
                    ({new Date(item.timestamp).toLocaleString()})
                  </span>
                </div>
                <button
                  className={`expand-button ${expandedNewsId === item.news_id ? "expanded" : ""}`}
                  onClick={() => toggleNewsExpand(item.news_id)}
                  aria-label={expandedNewsId === item.news_id ? "expand" : ""}
                >
                </button>
              </div>

              {expandedNewsId === (item.news_id || `news-${index}`) && (
                <div className="related-cards-container">
                  <h3>Related Market Events</h3>
                  {relatedCards[item.news_id || `news-${index}`] ? (
                    relatedCards[item.news_id || `news-${index}`].length > 0 ? (
                      <div className="cards-grid">
                        {relatedCards[item.news_id || `news-${index}`].map(card => (
                          <div
                            key={card.card_id}
                            className="related-card"
                            onClick={() => navigateToCardDetails(card.card_id)}
                          >
                            <h4>{card.question}</h4>
                            <div className="mini-chart">
                              {cardHistory[card.card_id] && (
                                <TimeSeriesChart
                                  data={cardHistory[card.card_id]}
                                  miniChart={true}
                                />
                              )}
                            </div>
                            <div className="card-options">
                              {card.options.map((option, idx) => (
                                <div className="option-row" key={idx}>
                                  <span>{option.option}</span>
                                  <span>{option.percentage}%</span>
                                </div>
                              ))}
                            </div>
                            <div className="see-more">See more</div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p>No related market events found</p>
                    )
                  ) : (
                    <p>Loading related events...</p>
                  )}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default NewsPage;
