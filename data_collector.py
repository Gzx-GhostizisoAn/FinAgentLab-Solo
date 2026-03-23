import yfinance as yf
import pandas as pd
import numpy as np
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from nlp_processor import process_news_data


SECTOR_STANDARD = {
    "XLK": "科技行业",
    "XLF": "金融行业",
    "XLE": "能源行业",
    "XLV": "医疗健康行业",
    "XLY": "可选消费行业",
    "XLP": "必需消费行业",
    "XLI": "工业行业",
    "XLU": "公用事业行业",
    "XLB": "原材料行业",
    "XLC": "通信服务行业"
}


PERIOD_MAP = {
    "短期": "1y",   
    "中期": "3y",    
    "长期": "5y"     
}


async def pull_standard_data(entity_type, entity_code, time_horizon, pull_news=False):

    period = PERIOD_MAP.get(time_horizon, "1y")
    result = {"success": True, "entity_type": entity_type, "time_horizon": time_horizon}

  
    if entity_type == "listed_company":
        company_data = await _pull_listed_company_standard(entity_code, period, pull_news)
        result.update(company_data)

  
    elif entity_type == "sector":
        sector_data = await _pull_sector_standard(entity_code, period, pull_news)
        result.update(sector_data)

 
    elif entity_type == "market":
        market_data = await _pull_market_standard(period)
        result.update(market_data)

    else:
        result = {"success": False, "error": "不支持的主体类型"}

    return result


async def _pull_listed_company_standard(code, period, pull_news):
    try:
        ticker = yf.Ticker(code)
        result = {"entity_name": code, "entity_code": code}

      
        price_df = ticker.history(period=period)[["Open", "High", "Low", "Close", "Volume"]].reset_index()
        price_df["Date"] = price_df["Date"].dt.normalize()  
        result["price_data"] = price_df.to_dict("records")
        result["data_count"] = len(price_df)

        tech_df = price_df[["Date", "Close"]].copy()
     
        tech_df["MA5"] = tech_df["Close"].rolling(5).mean()
        tech_df["MA20"] = tech_df["Close"].rolling(20).mean()
     
        tech_df["daily_return"] = tech_df["Close"].pct_change()
    
        tech_df["volatility_20d"] = tech_df["daily_return"].rolling(20).std()
   
        delta = tech_df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        tech_df["RSI_14"] = 100 - (100 / (1 + rs))
     
        result["tech_indicators"] = tech_df.dropna().to_dict("records")

  
        financial_result = {}
        try:
         
            quarterly_fin = ticker.quarterly_financials.T.sort_index()
       
            financial_result["total_revenue"] = quarterly_fin.get("Total Revenue", pd.Series()).to_dict()
            financial_result["net_income"] = quarterly_fin.get("Net Income", pd.Series()).to_dict()
            financial_result["gross_profit"] = quarterly_fin.get("Gross Profit", pd.Series()).to_dict()
     
            financial_result["revenue_growth_yoy"] = (quarterly_fin["Total Revenue"].pct_change(4) * 100).to_dict()
            financial_result["net_income_growth_yoy"] = (quarterly_fin["Net Income"].pct_change(4) * 100).to_dict()
      
            info = ticker.info
            financial_result["company_name"] = info.get("longName", code)
            financial_result["sector"] = info.get("sector", "未知行业")
            result["financial_data"] = financial_result
            result["entity_name"] = financial_result["company_name"]
        except Exception as e:
            result["financial_data"] = {"error": f"财务数据获取失败：{str(e)}"}

        if pull_news:
            news_list = await _get_news_data(code, financial_result.get("company_name", code))
            if news_list:
              
                news_result = process_news_data(news_list)
                result["news_data"] = news_result
            else:
                result["news_data"] = {"error": "未获取到相关新闻"}

        return result

    except Exception as e:
        return {"success": False, "error": f"上市公司数据拉取失败：{str(e)}"}


async def _pull_sector_standard(etf_code, period, pull_news):
    try:
        if etf_code not in SECTOR_STANDARD:
            return {"success": False, "error": "不支持的行业代码"}
        
        result = {
            "entity_name": SECTOR_STANDARD[etf_code],
            "entity_code": etf_code
        }
        sector_ticker = yf.Ticker(etf_code)
        spy_ticker = yf.Ticker("SPY") 

   
        sector_price = sector_ticker.history(period=period)[["Open", "High", "Low", "Close", "Volume"]].reset_index()
        sector_price["Date"] = sector_price["Date"].dt.normalize()
  
        spy_price = spy_ticker.history(period=period)[["Close", "Volume"]].reset_index()
        spy_price.columns = ["Date", "SPY_Close", "SPY_Volume"]
        spy_price["Date"] = spy_price["Date"].dt.normalize()
       
        merged_df = pd.merge(sector_price, spy_price, on="Date", how="left")
        result["price_data"] = merged_df.to_dict("records")
        result["data_count"] = len(merged_df)

   
        tech_df = merged_df[["Date", "Close"]].copy()
        tech_df["MA5"] = tech_df["Close"].rolling(5).mean()
        tech_df["MA20"] = tech_df["Close"].rolling(20).mean()
        tech_df["daily_return"] = tech_df["Close"].pct_change()
        tech_df["volatility_20d"] = tech_df["daily_return"].rolling(20).std()
     
        delta = tech_df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        tech_df["RSI_14"] = 100 - (100 / (1 + rs))
        result["tech_indicators"] = tech_df.dropna().to_dict("records")

     
        merged_df["sector_daily_return"] = merged_df["Close"].pct_change()
        merged_df["spy_daily_return"] = merged_df["SPY_Close"].pct_change()
       
        merged_df["return_vs_spy"] = merged_df["sector_daily_return"] - merged_df["spy_daily_return"]
 
        merged_df["cumulative_return_vs_spy"] = merged_df["return_vs_spy"].cumsum()
        result["relative_strength"] = merged_df[["Date", "return_vs_spy", "cumulative_return_vs_spy"]].dropna().to_dict("records")

   
        merged_df["volume_ma20"] = merged_df["Volume"].rolling(20).mean()
        merged_df["volume_change_rate"] = merged_df["Volume"].pct_change()
        merged_df["volume_vs_spy"] = merged_df["Volume"] / merged_df["SPY_Volume"]
        result["volume_trend"] = merged_df[["Date", "Volume", "volume_ma20", "volume_change_rate", "volume_vs_spy"]].dropna().to_dict("records")

    
        if pull_news:
            news_list = await _get_news_data(etf_code, SECTOR_STANDARD[etf_code])
            if news_list:
                news_result = process_news_data(news_list)
                result["news_data"] = news_result
            else:
                result["news_data"] = {"error": "未获取到相关行业新闻"}

        return result

    except Exception as e:
        return {"success": False, "error": f"行业数据拉取失败：{str(e)}"}


async def _pull_market_standard(period):
    try:
        result = {
            "entity_name": "整体市场",
            "entity_code": "SPY+QQQ+VIX+DXY+TNX"
        }

      
        spy = yf.Ticker("SPY").history(period=period)[["Open", "High", "Low", "Close", "Volume"]].reset_index()
        spy["Date"] = spy["Date"].dt.normalize()
   
        qqq = yf.Ticker("QQQ").history(period=period)[["Close"]].reset_index()
        qqq.columns = ["Date", "QQQ_Close"]
        qqq["Date"] = qqq["Date"].dt.normalize()
    
        vix = yf.Ticker("^VIX").history(period=period)[["Close"]].reset_index()
        vix.columns = ["Date", "VIX"]
        vix["Date"] = vix["Date"].dt.normalize()
      
        dxy = yf.Ticker("DX-Y.NYB").history(period=period)[["Close"]].reset_index()
        dxy.columns = ["Date", "Dollar_Index"]
        dxy["Date"] = dxy["Date"].dt.normalize()
       
        tnx = yf.Ticker("^TNX").history(period=period)[["Close"]].reset_index()
        tnx.columns = ["Date", "10Y_Treasury_Yield"]
        tnx["Date"] = tnx["Date"].dt.normalize()

     
        merged_df = spy.merge(qqq, on="Date", how="left")
        merged_df = merged_df.merge(vix, on="Date", how="left")
        merged_df = merged_df.merge(dxy, on="Date", how="left")
        merged_df = merged_df.merge(tnx, on="Date", how="left")
        result["market_core_data"] = merged_df.to_dict("records")
        result["data_count"] = len(merged_df)

    
        market_indicators = merged_df[["Date", "Close"]].copy()
  
        market_indicators["MA5"] = market_indicators["Close"].rolling(5).mean()
        market_indicators["MA20"] = market_indicators["Close"].rolling(20).mean()
        market_indicators["MA60"] = market_indicators["Close"].rolling(60).mean()
   
        market_indicators["daily_return"] = market_indicators["Close"].pct_change()
        market_indicators["volatility_20d"] = market_indicators["daily_return"].rolling(20).std()

        delta = market_indicators["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        market_indicators["RSI_14"] = 100 - (100 / (1 + rs))
    
        market_indicators["advance_decline_line"] = np.where(market_indicators["daily_return"] > 0, 1, -1).cumsum()

        vix_series = merged_df["VIX"].dropna()
        merged_df["VIX_history_percentile"] = vix_series.apply(lambda x: (vix_series <= x).mean() * 100)
        
        result["market_indicators"] = market_indicators.dropna().to_dict("records")
        result["vix_analysis"] = merged_df[["Date", "VIX", "VIX_history_percentile"]].dropna().to_dict("records")
        result["macro_indicators"] = merged_df[["Date", "Dollar_Index", "10Y_Treasury_Yield"]].dropna().to_dict("records")

        return result

    except Exception as e:
        return {"success": False, "error": f"整体市场数据拉取失败：{str(e)}"}


async def _get_news_data(keyword, name):
    session = aiohttp.ClientSession()
    news_list = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    try:
       
        sina_url = f"https://search.sina.com.cn/?q={name}+{keyword}&c=news&sort=time"
        async with session.get(sina_url, headers=headers, timeout=10) as resp:
            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")
            for item in soup.find_all("div", class_="box-result clearfix")[:20]:
                title_elem = item.find("h2")
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    content_elem = item.find("p", class_="content")
                    content = content_elem.get_text(strip=True) if content_elem else title
                    date_elem = item.find("span", class_="fgray_time")
                    date = date_elem.get_text(strip=True) if date_elem else ""
                    news_list.append({"title": title, "content": content, "date": date, "source": "新浪财经"})
        # 添加延迟以降低请求率
        await asyncio.sleep(60)
    except:
    
        try:
            eastmoney_url = f"https://so.eastmoney.com/news/s?keyword={name}+{keyword}&pagesize=20"
            async with session.get(eastmoney_url, headers=headers, timeout=10) as resp:
                html = await resp.text()
                soup = BeautifulSoup(html, "html.parser")
                for item in soup.find_all("div", class_="news-item")[:20]:
                    title_elem = item.find("a", class_="title")
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        content_elem = item.find("div", class_="content")
                        content = content_elem.get_text(strip=True) if content_elem else title
                        date_elem = item.find("span", class_="date")
                        date = date_elem.get_text(strip=True) if date_elem else ""
                        news_list.append({"title": title, "content": content, "date": date, "source": "东方财富"})
        except:
            pass
    finally:
        await session.close()

    return news_list