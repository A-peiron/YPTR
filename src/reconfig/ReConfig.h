/**
 * 这是我自己写的一些方法和类
 * 
 * 
 * 
*/
#ifndef RR_CONFIG_H_
#define RR_CONFIG_H_
#include <string>
#include <map>
#include "types.h"

struct TimeRange {
    int startHour;
    int startMinute;
    int endHour;
    int endMinute;
};

namespace rr
{
	class RrConfig
	{
	public:
		RrConfig()
		{
		}
		~RrConfig()
		{
		}
		bool ReadConfig(const std::string & filename);
		std::string ReadString(const char* section, const char* item, const char* default_value);
        int ReadInt(const char *section, const char *item, const int &default_value);
        float ReadFloat(const char* section, const char* item, const float& default_value);
		//这个是处理我自己特殊定义的配置使用的
		std::map<std::string, double> ReadStringToMap(const char *section, const char *item);
        void readPoints(std::string filename, Polygon &g_ploygon, int width, int height);

        std::vector<TimeRange> ReadStringToTimeRanges(const char *section, const char *item, const char* default_value);

    private:
        bool IsSpace(char c);
		bool IsCommentChar(char c);
		void Trim(std::string & str);
		bool AnalyseLine(const std::string & line, std::string& section, std::string & key, std::string & value);

	private:
		//std::map<std::string, std::string> settings_;
		std::map<std::string, std::map<std::string, std::string> >settings_;
	};
}


// 用来判断当前时间是否在规定时间之内
bool IsCurrentTimeInRange(int startHour, int startMinute, int endHour, int endMinute);
// 用来判断时间是否在多个时间段内
bool IsCurrentTimeInRanges(const std::vector<TimeRange>& timeRanges);
#endif


