# AI-2-Project

To download the data:

```bash 
git clone https://github.com/evankozliner/ScraXBRL.git
cd ScraXBRL 
python main.py
```

Try out a basic regression on amazon data to see how the extractor is used:

```bash
python correlation_model.py
```

  Basically the extractor joins the stock data and fundamentals data into a pandas dataframe broken into dates that work as time steps for usage in our models. Still need to do more research to find out what features will be best, but this should be enough info to get started on building temporal models.


## Added data as numpy array

Note that because it has been cast to a numpy array, columns names are lost. We are interested in the prediction of the 3rd column (amznclose)

```python
data = np.load('data.npy')
y_data = data[:,3]
```



