CREATE TABLE IF NOT EXISTS videos (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  file_name VARCHAR(255) NOT NULL,
  file_path TEXT,
  duration_sec DOUBLE NOT NULL,
  metadata_json JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS video_segments (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  video_id BIGINT NOT NULL,
  start_sec DOUBLE NOT NULL,
  end_sec DOUBLE NOT NULL,
  modality VARCHAR(32) NOT NULL,
  indexing_method VARCHAR(32) NOT NULL,
  content TEXT NOT NULL,
  artifact_json JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (video_id) REFERENCES videos(id)
);

-- Adjust the vector dimension to match the embedding model.
-- gemini-embedding-001 currently returns 3072 dimensions.
CREATE TABLE IF NOT EXISTS segment_embeddings (
  segment_id BIGINT PRIMARY KEY,
  embedding VECTOR(3072),
  embedding_model VARCHAR(128) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (segment_id) REFERENCES video_segments(id)
);
