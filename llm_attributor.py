import os
import json
from typing import Dict, Any
from openai import OpenAI

class LLMAttributor:

    
    def __init__(self):
       
        self.api_key = os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("请设置环境变量 QWEN_API_KEY")
        
      
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
       
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.model = "qwen-max"  # Qwen3 Max模型
    
  
    def _build_attribution_prompt(self, entity_info: Dict[str, Any], 
                                   prediction_result: Dict[str, Any],
                                   feature_importance: list,
                                   raw_data_summary: Dict[str, Any]) -> str:
        """
        构建专业的金融风险归因Prompt
        """
        prompt = f"""你是一位资深的金融风险分析师，拥有10年以上的金融市场分析经验。
请基于以下信息，生成一份专业、严谨、结构化的金融风险归因报告。

【基本信息】
- 预测主体：{entity_info.get('name', '未知')}
- 主体类型：{entity_info.get('type', '未知')}
- 主体代码：{entity_info.get('code', '未知')}
- 分析时间：{raw_data_summary.get('date_range', '未知')}

【风险预测结果】
- 最新风险概率：{prediction_result.get('latest_risk_probability', 0)}%
- 风险等级：{prediction_result.get('latest_risk_level', '未知')}
- 风险预测：{'存在风险' if prediction_result.get('latest_risk_prediction', 0) == 1 else '无风险'}

【关键特征重要性（TOP 10）】
{json.dumps(feature_importance, ensure_ascii=False, indent=2)}

【数据摘要】
- 数据点数：{raw_data_summary.get('data_count', 0)} 条
- 最新收盘价：{raw_data_summary.get('latest_close', '未知')}
- 最新波动率：{raw_data_summary.get('latest_volatility', '未知')}
- 市场情感：{raw_data_summary.get('sentiment', '未知')}

【报告要求】
请严格按照以下JSON格式输出，不要包含任何其他文字说明：
{{
  "executive_summary": "200字以内的风险归因摘要，专业、精炼",
  "risk_level_assessment": {{
    "current_level": "{prediction_result.get('latest_risk_level', '未知')}",
    "probability": {prediction_result.get('latest_risk_probability', 0)},
    "assessment": "对当前风险等级的专业评估"
  }},
  "key_risk_drivers": [
    {{
      "factor": "特征名称",
      "impact": "高/中/低",
      "explanation": "100字以内的专业解释，说明该特征如何影响风险"
    }}
  ],
  "market_context": "当前市场环境分析，150字以内",
  "mitigation_suggestions": [
    "具体、可操作的风险缓释建议1（100字以内）",
    "具体、可操作的风险缓释建议2（100字以内）",
    "具体、可操作的风险缓释建议3（100字以内）"
  ],
  "outlook": "对未来走势的专业展望，150字以内"
}}

请确保：
1. 所有分析基于提供的数据，不编造信息
2. 使用专业金融术语，但保持可读性
3. 建议具体、可操作，避免空泛
4. 严格JSON格式，不要有任何语法错误
"""
        return prompt
    
   
    def generate_attribution(self, entity_info: Dict[str, Any],
                            prediction_result: Dict[str, Any],
                            feature_importance: list,
                            raw_data_summary: Dict[str, Any]) -> Dict[str, Any]:
       
        try:
           
            prompt = self._build_attribution_prompt(
                entity_info, prediction_result, feature_importance, raw_data_summary
            )
            
           
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的金融风险分析师，擅长基于数据进行风险归因分析。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,  # 适度创造性
                max_tokens=2000,
                response_format={"type": "json_object"}  # 强制JSON输出
            )
            
           
            result_text = response.choices[0].message.content.strip()
            
          
            try:
                attribution_result = json.loads(result_text)
                return {
                    "success": True,
                    "attribution": attribution_result,
                    "model_used": self.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            except json.JSONDecodeError as e:
               
                import re
                json_match = re.search(r'\{[\s\S]*\}', result_text)
                if json_match:
                    attribution_result = json.loads(json_match.group())
                    return {
                        "success": True,
                        "attribution": attribution_result,
                        "model_used": self.model,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    }
                else:
                    return {
                        "success": False,
                        "error": f"JSON解析失败：{str(e)}",
                        "raw_response": result_text
                    }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"LLM调用失败：{str(e)}"
            }