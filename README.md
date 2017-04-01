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



