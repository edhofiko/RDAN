import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stemmer = StemmerFactory().create_stemmer()
dic = {"[SOS]":0, "[EOS]":1, "[NUMBER]":2}

class DataPrep:
	#Static Func
	map_lower = lambda x : x.lower()
	map_number = lambda x : re.sub("([0-9]+,[0-9]+)|([0-9]+)", " -NUMBER- ", x)
	map_connect = lambda x : re.sub(" - ", " ", x)
	map_non_alpha_stop = lambda x : re.sub("[^A-Za-z0-9.?!\[\]-]", " ", x)
	map_sos_eos = lambda x : re.sub("[.?!]", " -EOS- -- -SOS- ", x)
	map_add_sos_eos = lambda x : "-SOS- "+x+" -EOS-"
	map_rm_sos_eos = lambda x : re.sub("---SOS-( )*-EOS-", "", stemmer.stem(x))
	map_stem = lambda x : stemmer.stem(x)
	series_map = lambda series, map : series.astype(str).map(map)

	def casefold(series):
		return DataPrep.series_map(series, lambda x : DataPrep.map_lower(x))

	def mnemonic(series):
		return DataPrep.series_map(series, lambda x : DataPrep.map_rm_sos_eos(DataPrep.map_add_sos_eos(DataPrep.map_sos_eos(DataPrep.map_non_alpha_stop(DataPrep.map_connect(DataPrep.map_number(x)))))))

	def prep(series):
		return DataPrep.mnemonic(DataPrep.casefold(series))

	def chunks_op(chunks, ops):
		c = []
		for chunk in chunks:
			for op in ops:
				chunk = op(chunk)
			c.append(chunk)
		return pd.concat(c)

