import React, { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation, useParams } from 'react-router-dom';
import axios from "axios";
import NewsPage from "./NewsPage";
import TimeSeriesChart from "./TimeSeriesChart";
import './App.css';

const API_BASE_URL = window.location.hostname === "localhost"
  ? "http://localhost:5000"
  : `http://${window.location.hostname}:8000`;

const API_CARDS_URL = `${API_BASE_URL}/api/cards`;
const API_TAGS_URL = `${API_BASE_URL}/api/tags`;
const API_VOTE_HISTORY_URL = `${API_BASE_URL}/api/vote_history`;

export const HistoryChart = ({ cardId }) => {
  const [data, setData] = useState([]);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await axios.get(`${API_VOTE_HISTORY_URL}/${cardId}`);
        const formattedData = response.data.map(entry => ({
          timestamp: entry.timestamp,
          ...entry.votes,
        }));
        setData(formattedData);
      } catch (error) {
        console.error("Error fetching vote history:", error);
      }
    };
    fetchHistory();
  }, [cardId]);

  return <TimeSeriesChart data={data} />;
};

function App() {
  return (
    <>
      <Helmet>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </Helmet>
      <Router>
        <div className="app-container">
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
    <header className="app-header">
      <div className="header-left">
        <h1 className="app-title">Openmarket</h1>
        {location.pathname !== "/news" && (
          <button className="news-button" onClick={() => navigate('/news')}>
            News
          </button>
        )}
      </div>
      <div className="horizontal-bar"></div>
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
            {tags.map((tag) => (
              <option key={tag} value={tag}>
                {tag}
              </option>
            ))}
          </select>
        </div>
      )}
      <Routes>
        <Route path="/" element={<CardList cards={cards} />} />
        <Route path="/details/:card_id" element={<CardDetails cards={cards} />} />
        <Route path="/news" element={<NewsPage />} />
      </Routes>
    </>
  );
}

function CardList({ cards }) {
  const navigate = useNavigate();

  return (
    <div className="card-container">
      {cards.map((card) => (
        <div
          className="card"
          key={card.card_id}
          onClick={() => navigate(`/details/${card.card_id}`)}
        >
          <h3 className="card-title">{card.question}</h3>
          <div className="menu">
            {card.options.map((option, idx) => (
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

function CardDetails({ cards }) {
  const { card_id } = useParams();
  const card = cards.find((c) => c.card_id === card_id);

  const [selectedOption, setSelectedOption] = useState(null);

  const handleVote = () => {
    if (!selectedOption) return;

    axios.post(`${API_BASE_URL}/api/vote`, {
      card_id: card.card_id,
      option: selectedOption.option,
    })
      .then(() => {
        alert("Vote recorded successfully!");
        window.location.reload();
      })
      .catch((error) => {
        console.error("Error recording vote:", error);
        alert("Failed to record vote. Please try again.");
      });
  };

  if (!card) {
    return <div>Card not found</div>;
  }

  const handleOptionClick = (option) => {
    setSelectedOption(option);
  };

  return (
    <div className="details-page">
      <div className="left-card">
        <h2>{card.question}</h2>
        <div className="options-table">
          {card.options.map((option, idx) => (
            <div
              className="option-row clickable-option"
              key={idx}
              onClick={() => handleOptionClick(option)}
            >
              <span className="option-label">{option.option}</span>
              <span className="option-percentage">{option.percentage}%</span>
            </div>
          ))}
        </div>
      </div>

      <div className="right-card">
        <h2>Vote Option</h2>
        {selectedOption ? (
          <div className="vote-module">
            <p>Selected: <strong>{selectedOption.option}</strong></p>
            <p>Chance: <strong>{selectedOption.percentage}%</strong></p>
            <button className="vote-button" onClick={handleVote}>Vote</button>
          </div>
        ) : (
          <p>Please select an option to vote.</p>
        )}
        <div className="history-chart">
          <HistoryChart cardId={card_id} />
        </div>
      </div>
    </div>
  );
}

export default App;
