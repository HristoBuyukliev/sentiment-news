How to setup:

# Install requirements:
`pip install -r requirements.txt`
`mkdir data`
# Save the amazon data, the lexicon data, and the embeding data there.
# Convert data from the weird amazon format to csv:
`python transform_data.py`

# Train the models:
`python bow.py`
`python bilstm.py`
`python attention.py`
`python capsule.py`

# Train ensemble model and score on test:
`python ensemble.py`