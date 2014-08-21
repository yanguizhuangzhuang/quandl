# this is a util file used by other python scripts
import Quandl
import pandas
import statsmodels.api as sm
import string

_HOME='/home/lishaoyi'
_CSV_DIR=_HOME+'/data/csv'
_TXT_DIR=_HOME+'/data/txt'

with open ("/home/lishaoyi/data/txt/quandl-key.txt", "r") as f:
  auth_key = f.readline().rstrip('\n');

def price2return(df, attr):
  df[attr] = 100*(df[attr] - df[attr].shift(1))/df[attr]
  return df

def getDailyFromQuandl(ticker, src = 'NYSEARCA'):
  return Quandl.get('GOOG/' + src + '_'+ticker, authtoken=auth_key)

def getDailyFromQuandlAmex(ticker):
  return Quandl.get('GOOG/AMEX_'+ticker, authtoken=auth_key)

def getDailyFromCsv(ticker):
  return pandas.read_csv(_CSV_DIR+'/daily/'+ticker+'.csv', 
                         sep='|', index_col=[0], parse_dates=True)

def normalizeAttrName(prefix, attr):
  return string.replace((prefix+'_'+attr), ' ', '_')

def getOnlyColumn(df, attr, attr_n):
  return df[[attr]].rename(columns={attr:attr_n})

def regr(ticker1, ticker2, attr1, attr2):
  attr_1 = string.replace(attr1+"_1", ' ', '_')
  attr_2 = string.replace(attr1+"_2", ' ', '_')
  ticker1 = ticker1[[attr1]].rename(columns={attr1:attr_1})
  ticker2 = ticker2[[attr2]].rename(columns={attr2:attr_2})
  
  df = ticker1.join(other=ticker2, how='inner')
  fml = attr_1+' ~ '+attr_2
  return sm.ols(formula=fml, data=df).fit()

def multi2OneRegr(y, X, y_attr, X_attrs):
  y_attr_norm = normalizeAttrName('y', y_attr)
  df = price2return(getOnlyColumn(y, y_attr, y_attr_norm), y_attr_norm)
  
  for idx, a in enumerate(X_attrs):
    X_attrs[idx] = normalizeAttrName('x'+str(idx), a)
    tmp = price2return(getOnlyColumn(X[idx], a, X_attrs[idx]), X_attrs[idx])
    df = df.join(other = tmp, how = 'inner')
  df = df[1:]
  
  return sm.OLS(df[y_attr_norm], sm.add_constant(df[X_attrs]))

def compressDataframes(y, X, y_attr, X_attrs):
  y_attr_norm = normalizeAttrName('y', y_attr)
  df = price2return(getOnlyColumn(y, y_attr, y_attr_norm), y_attr_norm)
  
  for idx, a in enumerate(X_attrs):
    X_attrs[idx] = normalizeAttrName('x'+str(idx), a)
    tmp = price2return(getOnlyColumn(X[idx], a, X_attrs[idx]), X_attrs[idx])
    df = df.join(other = tmp, how = 'inner')
  df = df[1:]
  return df, y_attr_norm, X_attrs

 
def getCloseAttrName(df):
  attrs = df.columns.tolist()
  if 'Adjust Close' in attrs:
    return 'Adjust Close'
  elif 'Close' in attrs:
    return 'Close'
  return ''

def regrTask(y_ticker):
 X_tickers = open(_TXT_DIR+'/statarb-etfs.txt').read().splitlines()
 X = [None]*len(X_tickers)
 X_attrs = [None]*len(X_tickers)

 for i, t in enumerate(X_tickers):
   print t + ' ' + str(i) 
   X[i] = getDailyFromCsv(t)
   X_attrs[i] = getCloseAttrName(X[i])

 y = getDailyFromCsv(y_ticker)
 y_attr = getCloseAttrName(y)

 return multi2OneRegr(y, X, y_attr, X_attrs)

def getAllDataframeAndAttri(y_ticker):
  X_tickers = open(_TXT_DIR+'/statarb-etfs.txt.all').read().splitlines()
  X = [None]*len(X_tickers)
  X_attrs = [None]*len(X_tickers)

  for i, t in enumerate(X_tickers):
    print t + ' ' + str(i) 
    X[i] = getDailyFromCsv(t)
    X_attrs[i] = getCloseAttrName(X[i])

  y = getDailyFromCsv(y_ticker)
  y_attr = getCloseAttrName(y)
  return y, X, y_attr, X_attrs  

def getAllDataframeAndAttriCompressed(y_ticker):
  y, X, y_attr, X_attrs = getAllDataframeAndAttri(y_ticker)
  return compressDataframes(y, X, y_attr, X_attrs)

def regrFromDataframe(df, y_attr, X_attrs, date, n_aux, n_mfac):
  df = getLookbackPeriod(df, date, n_mfac)

  model_mfac = sm.OLS(df[y_attr], sm.add_constant(df[X_attrs])).fit()
  beta_mfac  = model_mfac.params
  resid_mfac = model_mfac.resid

  aux = calcAuxilaryArray(resid_mfac, n_aux)
  aux_res = sm.OLS(aux[1:], sm.add_constant(aux[:-1])).fit()
    
  return calcSScore(aux_res.params, aux_res.resid, aux[-1]), beta_mfac, aux_res.params

def regrTaskNdays(y_ticker, date = '2014-02-29', n_aux = 60, n_mfac = 90):
  df, y_attr, X_attrs = getAllDataframAndAttriCompressed(y_ticker)
  return regrFromDataframe(df, y_attr, X_attrs, date, n_aux, n_mfac)

def regrTaskNdaysAll(ticker, n_mfac = 90, n_aux = 60):
    df, y_attr, X_attrs = getAllDataframeAndAttriCompressed(ticker)
    arr = []
    for dt in df.index[n_mfac:]:
      dt_str = dt.strftime('%Y-%m-%d')
      print dt_str
      s, beta_mfac, beta_aux = regrFromDataframe(df, y_attr, X_attrs, dt_str, n_aux, n_mfac)
      arr.append((s, beta_mfac, beta_aux))
    bs = [r[2][1] for r in arr]
    return arr, df, sum(bs)/len(bs)

def calcAuxilaryArray(resid_mfac, n_aux):
  aux = [0]*n_aux
  a = 0
  for i, r in enumerate(resid_mfac[-n_aux:]):
    aux[i] = a = a + r
  return aux

def calcSScoreAgainstExisting(beta_mfac, beta_aux, df, n_aux):
  y = df.iloc[:, 0]
  X = df.iloc[:, 1:]
  resid_mfac = y - sm.add_constant(X).dot(beta_mfac)
  aux = calcAuxilaryArray(resid_mfac, n_aux)

  resid_aux = aux[1:] - sm.add_constant(aux[:-1]).dot(beta_aux)
  return calcSScore(beta_aux, resid_aux, aux[-1])

def calcSScoreAgainstExistingAll(fitted_res, day_of_beta, df, n_mfac, n_aux, n_forecast):
  dummy_s, beta_mfac, beta_aux = fitted_res[day_of_beta]
  arr = []
  for dt in df.index[n_mfac+day_of_beta:n_mfac+day_of_beta+n_forecast]:
    dt_str = dt.strftime('%Y-%m-%d')
#    print dt_str
    window = getLookbackPeriod(df, dt_str, n_mfac)
    s = calcSScoreAgainstExisting(beta_mfac, beta_aux, window, n_aux)
    arr.append(s)
  return arr

def calcSScore(beta, resid, xt):
  a, b = beta
  std = resid.std()
  m = a / (1 -b)
  sigma_eq = std / ((1 - b**2)**0.5)
  return (xt - m) / sigma_eq

def getLookbackPeriod(df, date, n):
  di = df.index.get_loc(pandas.to_datetime(date))
  return df[di - n + 1:di]  

def refreshDailyCsvFromQuandl():
  tickers = open(_TXT_DIR+'/statarb-etfs.txt.all').read().splitlines()

  for t in tickers:
    print "Processing request for " + t + "..."
    cacheDailyCsvFromQuandl(t)
  
def cacheDailyCsvFromQuandl(ticker, src='NYSEARCA'):
  getDailyFromQuandl(ticker, src).to_csv(_CSV_DIR+'/daily/'+ticker+'.csv', sep = '|')
