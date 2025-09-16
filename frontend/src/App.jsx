import React, { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation, useParams } from 'react-router-dom';
import axios from "axios";
import NewsPage from "./NewsPage";
import TimeSeriesChart from "./TimeSeriesChart";
import './App.css';
import CardDetails from "./CardDetails";

const API_BASE_URL = window.location.hostname === "localhost"
  ? "http://localhost:5002"
  : "";


const API_CARDS_URL = `${API_BASE_URL}/api/cards`;
const API_TAGS_URL = `${API_BASE_URL}/api/tags`;
const API_VOTE_HISTORY_URL = `${API_BASE_URL}/api/vote_history`;


export const HistoryChart = ({ cardId }) => {
  const [actualData, setActualData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        const actualResponse = await axios.get(`${API_VOTE_HISTORY_URL}/${cardId}`);
        const formattedActualData = actualResponse.data.map(entry => ({
          timestamp: entry.timestamp,
          ...entry.votes,
        }));
        setActualData(formattedActualData);
      } catch (error) {
        console.error("Error fetching chart data:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [cardId]);

  if (isLoading) {
    return <div>Loading chart data...</div>;
  }

  return <TimeSeriesChart data={actualData} showEstimated={false} />;
};


function App() {
  return (
    <>
      <Helmet>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </Helmet>
      <Router>
        <div className="App">
          <Header />
          <MainContent />
        </div>
      </Router>
    </>
  );
}

function Header() {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <header className="navigation">
      <div className="navigation-container">
        <div className="nav-logo">
          <button className="app-title" onClick={() => navigate('/')}>
            Openmarket
          </button>
        </div>
        
        <div className="desktop-nav">
          <div className="nav-links">
            {location.pathname !== "/news" && (
              <button className="news-button" onClick={() => navigate('/news')}>
                News
              </button>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

function MainContent() {
  const location = useLocation();
  const [cards, setCards] = useState([]);
  const [tags, setTags] = useState([]);
  const [selectedTag, setSelectedTag] = useState("");

  useEffect(() => {
    let url = API_CARDS_URL;
    if (selectedTag) {
      url += `?tag=${selectedTag}`;
    }

    axios.get(url)
      .then((response) => {
        setCards(response.data);
      })
      .catch((error) => console.error("Error fetching cards:", error));
  }, [selectedTag]);

  useEffect(() => {
    axios.get(API_TAGS_URL)
      .then((response) => {
        setTags(response.data);
      })
      .catch((error) => console.error("Error fetching tags:", error));
  }, []);

  return (
    <>
      {location.pathname === "/" && (
        <div className="tag-filter">
          <label htmlFor="tag-select">Filter by Tag: </label>
          <select
            id="tag-select"
            value={selectedTag}
            onChange={(e) => setSelectedTag(e.target.value)}
          >
            <option value="">All</option>
            {Array.isArray(tags) && tags.map((tag) => (
              <option key={tag} value={tag}>
                {tag}
              </option>
            ))}
          </select>
        </div>
      )}
      <Routes>
        <Route path="/" element={<CardList cards={cards} />} />
        <Route path="/details/:card_id" element={<CardDetails cards={cards} API_BASE_URL={API_BASE_URL} />} />
        <Route path="/news" element={<NewsPage />} />
      </Routes>
    </>
  );
}

function CardList({ cards }) {
  const navigate = useNavigate();

  return (
    <div className="card-container">
      {Array.isArray(cards) && cards.map((card) => (
        <div
          className="card"
          key={card.card_id}
          onClick={() => navigate(`/details/${card.card_id}`)}
        >
          <h3 className="card-title">{card.question}</h3>
          <div className="menu">
            {Array.isArray(card.options) && card.options.map((option, idx) => (
              <div className="menu-item" key={idx}>
                <span className="option-label">{option.option}</span>
                <span className="option-percentage">{option.percentage}%</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export default App;
