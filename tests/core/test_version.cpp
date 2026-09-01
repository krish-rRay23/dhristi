#include "drishti/core/config.h"
#include "drishti/core/export.h"
#include "drishti/core/version.h"

#include <gtest/gtest.h>

using namespace drishti::core;

TEST(VersionTest, VersionComponentsMatchConfig) {
    const auto v = version();
    EXPECT_EQ(v.major, DRISHTI_VERSION_MAJOR);
    EXPECT_EQ(v.minor, DRISHTI_VERSION_MINOR);
    EXPECT_EQ(v.patch, DRISHTI_VERSION_PATCH);
}

TEST(VersionTest, VersionStringIsNonEmpty) {
    const auto v = version();
    EXPECT_FALSE(v.str().empty());
    EXPECT_STREQ(v.str().data(), DRISHTI_VERSION_STRING);
}

TEST(VersionTest, BannerContainsProjectName) {
    const auto b = banner();
    EXPECT_NE(b.find(DRISHTI_VERSION_STRING), std::string::npos);
}

TEST(CoreTest, ProjectNameReturnsExpected) {
    EXPECT_STREQ(project_name(), "D\u1e5b\u1e63\u1e6di");
}
