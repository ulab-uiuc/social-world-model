import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import { HistoryChart } from './App';

const CardDetails = ({ cards, API_BASE_URL }) => {
  const { card_id } = useParams();
  const card = cards.find((c) => c.card_id === card_id);

  const [selectedOption, setSelectedOption] = useState(null);
  const [chartMode, setChartMode] = useState(1);
  const [reasons, setReasons] = useState([]);
  const [selectedReason, setSelectedReason] = useState(null);
  const [selectedModel, setSelectedModel] = useState('gpt-4o-mini');
  const [isGenerating, setIsGenerating] = useState(false);

  const availableModels = [
    { value: 'gpt-5', label: 'GPT-5 (Latest)' },
    { value: 'o4-mini', label: 'o4-mini' },
    { value: 'claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet' },
    { value: 'claude-3-5-haiku-20241022', label: 'Claude 3.5 Haiku' },
    { value: 'claude-3-opus-20240229', label: 'Claude 3 Opus' },
    { value: 'gpt-4o', label: 'GPT-4o' },
    { value: 'gpt-4o-mini', label: 'GPT-4o Mini' },
    { value: 'gpt-4', label: 'GPT-4' },
    { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' }
  ];

  const API_OPTION_REASONS_URL = `${API_BASE_URL}/api/option_reasons`;

  const fetchReasons = async (option, model) => {
    try {
      const response = await axios.get(`${API_OPTION_REASONS_URL}/${card.card_id}/${option.option}?model=${model}`);
      if (response.data.reasons && response.data.reasons.length > 0) {
        setReasons(response.data.reasons);
        setIsGenerating(false);
      } else {
        // No reasoning found, need to generate
        await generateReasons(option, model);
      }
    } catch (error) {
      console.error("Error fetching reasons:", error);
      setReasons([]);
      setIsGenerating(false);
    }
  };

  const generateReasons = async (option, model) => {
    setIsGenerating(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/generate_reasons`, {
        card_id: card.card_id,
        option: option.option,
        model: model
      });
      setReasons(response.data.reasons || []);
    } catch (error) {
      console.error("Error generating reasons:", error);
      setReasons([]);
      alert('Failed to generate reasons. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleOptionClick = async (option) => {
    setSelectedOption(option);
    setSelectedReason(null);
    await fetchReasons(option, selectedModel);
  };

  const handleModelChange = async (newModel) => {
    console.log('Model changed to:', newModel);
    setSelectedModel(newModel);
    if (selectedOption) {
      setSelectedReason(null);
      await fetchReasons(selectedOption, newModel);
    }
  };

  const handleReasonSelect = (reason) => {
    setSelectedReason(reason);
  };

  const handleVote = () => {
    if (!selectedOption || !selectedReason) return;

    axios.post(`${API_BASE_URL}/api/vote`, {
      card_id: card.card_id,
      option: selectedOption.option,
      reason_id: selectedReason.reason_id,
      model: selectedModel,
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

  return (
    <div className="details-page">
      <div className="left-card">
        <h2>{card.question}</h2>

        <div className="options-container">
          {card.options.map((option, idx) => (
            <div
              key={idx}
              className={`option-row clickable-option ${selectedOption && selectedOption.option === option.option ? 'selected-option' : ''}`}
              onClick={() => handleOptionClick(option)}
            >
              <span className="option-label">{option.option}</span>
              <span className="option-percentage">{option.percentage}%</span>
            </div>
          ))}
        </div>

        {selectedOption && (
          <div className="model-selection">
            <h4>AI Model:</h4>
            <select 
              value={selectedModel} 
              onChange={(e) => {
                e.preventDefault();
                e.stopPropagation();
                handleModelChange(e.target.value);
              }}
              className="model-selector"
              onClick={(e) => {
                e.stopPropagation();
                console.log('Select clicked');
              }}
            >
              {availableModels.map((model) => (
                <option key={model.value} value={model.value}>
                  {model.label}
                </option>
              ))}
            </select>
          </div>
        )}

        {selectedOption && isGenerating && (
          <div className="generating-message">
            <p>🤖 Generating reasoning with {availableModels.find(m => m.value === selectedModel)?.label}...</p>
          </div>
        )}

        {selectedOption && reasons.length > 0 && !isGenerating && (
          <div className="reasons-container">
            <h4>Select your reason:</h4>
            <div className="reasons-list">
              {reasons.map((reason) => (
                <div
                  key={reason.reason_id}
                  className={`reason-item ${selectedReason && selectedReason.reason_id === reason.reason_id ? 'selected' : ''}`}
                  onClick={() => handleReasonSelect(reason)}
                >
                  <p>{reason.text}</p>
                  <span className="reason-votes">({reason.votes} votes)</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="right-card">
        <h2>Vote Option</h2>
        {selectedOption ? (
          <div className="vote-module">
            <p>Selected: <strong>{selectedOption.option}</strong></p>
            <p>Chance: <strong>{selectedOption.percentage}%</strong></p>

            <button
              className="vote-button"
              onClick={handleVote}
              disabled={!selectedReason}
            >
              Vote
            </button>
          </div>
        ) : (
          <p>Please select an option to vote.</p>
        )}
        <div className="history-chart-container">
          <div className="chart-controls">
            <button
              className={`chart-mode-btn ${chartMode === 1 ? 'active' : ''}`}
              onClick={() => setChartMode(1)}
            >
              Real
            </button>
            <button
              className={`chart-mode-btn ${chartMode === 2 ? 'active' : ''}`}
              onClick={() => setChartMode(2)}
            >
              Estimated
            </button>
            <button
              className={`chart-mode-btn ${chartMode === 3 ? 'active' : ''}`}
              onClick={() => setChartMode(3)}
            >
              Both
            </button>
          </div>
          <div className="history-chart">
            <HistoryChart cardId={card_id} mode={chartMode} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default CardDetails;
